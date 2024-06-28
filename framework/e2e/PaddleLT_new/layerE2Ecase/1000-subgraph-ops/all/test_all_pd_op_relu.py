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
    class PrimitiveOp_378f3bd9c84853d2d2ff21b4db74f841(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9420e0791338c9d263ce4fa7913600b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_378f3bd9c84853d2d2ff21b4db74f841
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_202bfdc0ac9db07356cae4b693623ccc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83e7086ef33dc4a8d8d2ae6e72826b09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202bfdc0ac9db07356cae4b693623ccc
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.626307010650635, 4.391082763671875, 4.806268215179443, 4.401785850524902, 4.369851112365723, 4.688978672027588, 4.786715984344482, 4.2709174156188965, 4.522144794464111, 4.13695764541626, 5.095004558563232, 4.644773006439209, 3.8188412189483643, 4.205009937286377, 4.518152236938477, 4.431371212005615, 4.864565849304199, 3.6953768730163574]], dtype='float32').reshape([1, 18]),
            ]


    
    class PrimitiveOp_cd6460d9338b9f965d298e71d4ef198a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_110305d7875842fad1c2ff7a12621021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd6460d9338b9f965d298e71d4ef198a
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.27309513092041, 5.833820343017578, 5.503009796142578, 5.4817891120910645, 5.425642013549805, 6.097849369049072, 5.734645843505859, 5.922000885009766, 5.353610038757324, 5.361513137817383, 4.979347229003906, 6.077544212341309, 5.27100133895874, 5.504151821136475, 5.766942977905273, 5.257235527038574, 5.357554912567139, 5.529504299163818, 5.592182159423828, 5.393463611602783, 5.493752479553223, 6.035822868347168, 5.610565185546875]], dtype='float32').reshape([1, 23]),
            ]


    
    class PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_002c36452ffe5275fac93fbe54c63152(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8b15e4a9a7a2f200c08c3224be2704d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_002c36452ffe5275fac93fbe54c63152
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05dbfdec08ac3aa440da8bb491d33738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_80397bb364335531627c2a66568545dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_029d435b956a6ee5f45cf1e75c139312(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5f3e044367baedbd8635e511318b8eed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_580fdce3ad3f715cf8180bbaa34e68a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f3e044367baedbd8635e511318b8eed
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_580fdce3ad3f715cf8180bbaa34e68a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f3e044367baedbd8635e511318b8eed
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15c08707d8b252c03051dfd25790e353(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a66dfdf13cefdcd486fbf18b8f8b32de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.365582466125488]], [[7.731196403503418]], [[7.964057445526123]], [[6.985565662384033]], [[7.5866594314575195]], [[8.946654319763184]], [[7.98096227645874]], [[8.842916488647461]], [[7.250166893005371]], [[7.330620765686035]], [[7.9806976318359375]], [[8.2101411819458]], [[8.502208709716797]], [[7.358593940734863]], [[7.772006988525391]], [[7.207477569580078]], [[7.093987464904785]], [[8.185877799987793]], [[8.008213996887207]], [[7.6435675621032715]], [[7.370347499847412]], [[7.112212181091309]], [[7.433649063110352]], [[7.9769439697265625]], [[7.491349697113037]], [[8.110529899597168]], [[7.971155166625977]], [[8.05787467956543]], [[7.60438346862793]], [[7.9871745109558105]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_eb42587a18ca51b292408a36234475f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f23e5f65f2d81fcc0b69e8c24dfc3fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5db333f5f66836f46eb588c1df49017d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_983136a7f43dd9cb493796fddaf62415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc6091e804159ad162e931c1a2008c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab2f12719d96b296ec2577aa264e064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23ad7e6e07c047477bced54808fb774d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43049a2d47b310d3f875afeadaa157db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.194438934326172]], [[8.588522911071777]], [[7.182609558105469]], [[7.648037433624268]], [[6.894355773925781]], [[6.827334880828857]], [[7.831111431121826]], [[7.7364959716796875]], [[8.895010948181152]], [[7.1844940185546875]], [[7.992562294006348]], [[7.785795211791992]], [[8.145564079284668]], [[8.470157623291016]], [[8.22587776184082]], [[7.272141933441162]], [[8.16726303100586]], [[8.357715606689453]], [[7.874889850616455]], [[7.3341898918151855]], [[7.798108100891113]], [[7.923760414123535]], [[8.171854972839355]], [[8.929399490356445]], [[6.554434299468994]], [[7.777000427246094]], [[8.502005577087402]], [[8.291655540466309]], [[8.365804672241211]], [[8.238239288330078]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_8f0f2e543b7b4e396d74dfd2b77a0d24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8aa6a208551763b029a4175fcd015eae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b216429327e4ea6895aa7fea7d01d8ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa6a208551763b029a4175fcd015eae
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5952028036117554]], [[1.1798752546310425]], [[1.684241533279419]], [[1.4642938375473022]], [[1.7384600639343262]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    
    class PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61787f189c65810d0483d1da7a7e1c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.7241156101226807]], [[2.818718194961548]], [[3.44044828414917]], [[3.0282323360443115]], [[2.9832065105438232]], [[2.824711799621582]], [[3.593562126159668]], [[3.125070095062256]], [[2.591668128967285]], [[3.1199605464935303]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    
    class PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_abe57ba0e6e948b2d582cb46dc5d07c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.417937755584717]], [[5.54685640335083]], [[6.385406494140625]], [[6.424187660217285]], [[6.714924335479736]], [[5.881267547607422]], [[5.836642742156982]], [[5.545405864715576]], [[6.1585845947265625]], [[6.060319900512695]], [[5.8800458908081055]], [[6.286401748657227]], [[6.463059902191162]], [[5.401740074157715]], [[5.785524368286133]], [[5.819664478302002]], [[6.0124735832214355]], [[6.33400297164917]], [[6.509230613708496]], [[6.19317626953125]], [[6.521552085876465]], [[6.344600677490234]], [[5.961433410644531]], [[6.324902534484863]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_af725c2478af0ddf19ea19340e8c93ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9c08f695dcae0c302451c1485c63a54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a33ac0f0573f4de512ec370ae553528c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_854ecb0c9872d17a422de1b8b6f88e06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33ac0f0573f4de512ec370ae553528c
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68a3e1dda091b4707ad48ef454780e90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.443090915679932]], [[4.749655723571777]], [[3.8140125274658203]], [[4.799515247344971]], [[4.431580066680908]], [[4.186607837677002]], [[4.143815994262695]], [[4.519565582275391]], [[4.756207466125488]], [[4.297165393829346]], [[3.9445648193359375]], [[4.05826473236084]], [[4.993553638458252]], [[4.120225429534912]], [[4.165502071380615]], [[4.1748857498168945]], [[3.941878080368042]], [[4.543561935424805]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32ef9971183539453c98627db17e0d30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.8558502197265625]], [[6.249524116516113]], [[5.586520195007324]], [[5.772902965545654]], [[6.590846538543701]], [[6.154374122619629]], [[6.037872314453125]], [[5.574150085449219]], [[5.730854034423828]], [[5.262627124786377]], [[5.772827625274658]], [[6.190662384033203]], [[5.621458053588867]], [[5.770803451538086]], [[5.573040962219238]], [[5.884099960327148]], [[5.568533897399902]], [[5.506991863250732]], [[6.336510181427002]], [[5.539333820343018]], [[5.555763244628906]], [[6.508022308349609]], [[5.37600564956665]], [[5.509634971618652]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_bf7f5758103f05063fd84e11f80d2885(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77042b92df50b00438837a5dd73cf2a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81236129c333dfe7ae73bbcbb0979cbf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87cc7854a03421fec3c4193629f9bb6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81236129c333dfe7ae73bbcbb0979cbf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9399208426475525]], [[0.7881795167922974]], [[0.7001737952232361]], [[1.489819884300232]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_bf7f5758103f05063fd84e11f80d2885(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d6a79f19dffaf1a401b1a360fa95eb71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a1fb8c320dc1fceb2df3ca14e04d78c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a79f19dffaf1a401b1a360fa95eb71
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.274381637573242]], [[3.215075969696045]], [[2.314345121383667]], [[2.7561116218566895]], [[3.2190792560577393]], [[3.261092185974121]], [[2.7221601009368896]], [[2.923419237136841]], [[3.007906436920166]], [[3.288865566253662]], [[3.1404919624328613]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac26949d453892b38139ad28d5c946ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.629995346069336]], [[7.557254314422607]], [[8.062764167785645]], [[8.629494667053223]], [[7.266910552978516]], [[7.732415199279785]], [[7.328917980194092]], [[7.593316555023193]], [[7.870702743530273]], [[7.665899753570557]], [[8.408510208129883]], [[7.446771144866943]], [[8.450179100036621]], [[8.21670913696289]], [[8.057311058044434]], [[8.340054512023926]], [[8.768411636352539]], [[8.051828384399414]], [[8.499567985534668]], [[8.646636009216309]], [[7.833431720733643]], [[7.5960564613342285]], [[8.436894416809082]], [[8.195596694946289]], [[8.104879379272461]], [[7.728816032409668]], [[8.08309268951416]], [[7.9412055015563965]], [[8.761932373046875]], [[8.15397834777832]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e01901587f7c1f9983f1dce5c54611dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96bc1c049ccc40f991f92f5ff19250dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e01901587f7c1f9983f1dce5c54611dc
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_362e5f023bc49db46e3a4d7b3006fb6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6c2c7c96365c3bc4b6270266a7deb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.589501857757568]], [[3.9595940113067627]], [[4.664587020874023]], [[4.551024913787842]], [[4.109001636505127]], [[4.198220252990723]], [[4.432218551635742]], [[4.4817118644714355]], [[3.9227426052093506]], [[4.625372409820557]], [[3.8187661170959473]], [[4.21016788482666]], [[4.568661689758301]], [[4.094541072845459]], [[4.191646575927734]], [[4.210667133331299]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_a85c0077329ccb9dda1330c3c02626cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd8b45f170612ca85cf7c363acc6cf46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fc8c4ce0c5bd6f6ce43aa5925df1116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_797475bab3176cb964e76e9893d4aeff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b76b9bedeada57d3778ddc20bb5f69e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33ac0f0573f4de512ec370ae553528c
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_857c21fa392c6921805c4f02babeb3bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e97f41c56433b71b4a055491d5f0160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_857c21fa392c6921805c4f02babeb3bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00f240ae374f789ad46a36417fa42a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.331165313720703]], [[6.932826519012451]], [[6.909614086151123]], [[7.851978302001953]], [[7.19744348526001]], [[7.731747627258301]], [[6.7408857345581055]], [[7.960355281829834]], [[6.684189796447754]], [[7.306125164031982]], [[6.787485122680664]], [[6.810930252075195]], [[6.970196723937988]], [[7.652226448059082]], [[7.5926666259765625]], [[7.016533374786377]], [[6.701245307922363]], [[7.672969818115234]], [[7.558865547180176]], [[7.018603324890137]], [[7.665589332580566]], [[7.1204938888549805]], [[6.956162452697754]], [[7.111342906951904]], [[7.7293500900268555]], [[7.124806880950928]], [[7.486032962799072]], [[7.5760722160339355]], [[7.356403350830078]], [[7.904211521148682]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_7fe84f2534f9c690d4103c62c9fb6e84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b440e757310cef06249aecb8b8ba3a96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fe84f2534f9c690d4103c62c9fb6e84
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11a218c469172afe6b54a19cb7f70088(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_472f90195d81e0ba99d29b026a6f2719(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a218c469172afe6b54a19cb7f70088
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43e922670420b170cf250897d83df52e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.2826313972473145]], [[6.334243297576904]], [[6.943402290344238]], [[6.860344886779785]], [[7.050297737121582]], [[7.861888408660889]], [[6.671663284301758]], [[6.760298728942871]], [[7.774744510650635]], [[7.529026985168457]], [[8.103219032287598]], [[6.956748008728027]], [[7.525053024291992]], [[6.947916030883789]], [[6.68371057510376]], [[7.102306365966797]], [[6.944980144500732]], [[6.7735161781311035]], [[7.2324910163879395]], [[6.96896505355835]], [[7.399407386779785]], [[6.78544807434082]], [[8.230155944824219]], [[6.772273540496826]], [[7.684123516082764]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1eb387737cada12545b6f359998c1e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79f904f616c9ebf47192fc807364cd83(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f3b48a5cf3b3d8362be7788a53b147a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79f904f616c9ebf47192fc807364cd83
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c95165ffe6d3be08174c5deca025730c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f3e044367baedbd8635e511318b8eed
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c95165ffe6d3be08174c5deca025730c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f3e044367baedbd8635e511318b8eed
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c28cda091fe8fba6c9bc97e3dd6c576b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_595b65274641f25d6c1b5721129fc0d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_275931edcb33e42ae3bd2cc41d482b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_595b65274641f25d6c1b5721129fc0d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd11c87a67ce17237c6618fa09e6521d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.324487686157227]], [[5.197154998779297]], [[5.39374303817749]], [[4.502297401428223]], [[5.0500264167785645]], [[5.456243515014648]], [[4.709370136260986]], [[5.073709487915039]], [[5.223665714263916]], [[5.386110782623291]], [[4.320528507232666]], [[5.036959171295166]], [[5.214683532714844]], [[5.159992694854736]], [[5.648549556732178]], [[5.104668140411377]], [[5.089097499847412]], [[4.991738319396973]], [[4.9874982833862305]], [[4.739211559295654]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b440e757310cef06249aecb8b8ba3a96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fe84f2534f9c690d4103c62c9fb6e84
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80cafb8e351b9d2533a53b03169d914e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.766676902770996]], [[4.8825483322143555]], [[4.5546064376831055]], [[4.451656818389893]], [[4.954763889312744]], [[5.02230167388916]], [[4.508750915527344]], [[4.965527534484863]], [[4.956589698791504]], [[4.86350679397583]], [[4.453602313995361]], [[4.658884048461914]], [[4.835541725158691]], [[4.794821739196777]], [[4.3191351890563965]], [[4.349730968475342]], [[4.615575790405273]], [[5.2794270515441895]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_6e353866c8451354322f2cbb14fd90a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e01901587f7c1f9983f1dce5c54611dc
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9420e0791338c9d263ce4fa7913600b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_378f3bd9c84853d2d2ff21b4db74f841
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_654bac7a0255408cb0b178458d8f3adb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6a8716162cf0ee502dcb01428ee5481(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_93e122920071fba34ad52b992739a4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6a8716162cf0ee502dcb01428ee5481
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e32abaabbd1b469a97d528064c779d0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36ffd12e789b7189da2a6a80b8f30fe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ffd12e789b7189da2a6a80b8f30fe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e32abaabbd1b469a97d528064c779d0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ffd12e789b7189da2a6a80b8f30fe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ffd12e789b7189da2a6a80b8f30fe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71dbe82e5b5a0033f8aa37046566088a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc192a27a376203d9453b718f413122d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_018825a7223ba53769946dc2939208b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018825a7223ba53769946dc2939208b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5882a85a4e4c9827e661271da9790b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06a21d914a201b1d3c49cd95859913f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06a21d914a201b1d3c49cd95859913f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18ce6972ece5cbfc54f4f27c229f487e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f455eb9bc785110392d6bc24992c8ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f455eb9bc785110392d6bc24992c8ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18ce6972ece5cbfc54f4f27c229f487e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f455eb9bc785110392d6bc24992c8ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f455eb9bc785110392d6bc24992c8ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6917559ba914a1b695f53f325e327ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2aa47408bff3d6698fd431a056832f0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2aa47408bff3d6698fd431a056832f0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70c510b19b59da2e80c037419630d949(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6feb626794b0b4453c8ed402dc3e3940(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6feb626794b0b4453c8ed402dc3e3940(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_189690d114989cf31ecde125007ca5bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_972c33afd5c280af8ec71ea6976891e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_189690d114989cf31ecde125007ca5bc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275931edcb33e42ae3bd2cc41d482b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_595b65274641f25d6c1b5721129fc0d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_131128f4435bfce645b4ffe48b6819a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.547247409820557]], [[5.241771221160889]], [[4.631717205047607]], [[5.429015636444092]], [[4.983003616333008]], [[4.7412495613098145]], [[4.678460597991943]], [[5.05424165725708]], [[4.121486186981201]], [[4.833622932434082]], [[4.7639336585998535]], [[4.579180717468262]], [[4.328996181488037]], [[4.869755744934082]], [[5.027015686035156]], [[4.672824859619141]], [[4.301534652709961]], [[4.660597324371338]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_9f23e5f65f2d81fcc0b69e8c24dfc3fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5926c2960a31c0510fb4257d4d22c0d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.95380163192749]], [[5.899272918701172]], [[6.4619035720825195]], [[6.993582725524902]], [[6.350670337677002]], [[6.611702919006348]], [[6.624584197998047]], [[6.251131534576416]], [[5.933366775512695]], [[5.4472479820251465]], [[6.299964904785156]], [[6.762087345123291]], [[6.120079040527344]], [[6.5732316970825195]], [[5.486151218414307]], [[6.62627649307251]], [[6.923728942871094]], [[6.560196399688721]], [[6.101138591766357]], [[6.05964994430542]], [[7.280458450317383]], [[6.875477313995361]], [[5.495212078094482]], [[6.6487884521484375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_6a4c5bfd3d6afc713c0678b8f95a670a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37246ccbdc537cd962da9b1ff2917bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.828707695007324]], [[4.706035137176514]], [[4.936190605163574]], [[4.7714033126831055]], [[5.0137248039245605]], [[4.696411609649658]], [[4.654387950897217]], [[4.914733409881592]], [[4.297507286071777]], [[4.808157444000244]], [[4.754739284515381]], [[4.652151107788086]], [[4.769132614135742]], [[5.319541931152344]], [[5.285849094390869]], [[4.965315818786621]], [[5.107652187347412]], [[5.249327659606934]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_a4be20b6032c6b4bc2c02d0c0c5b3b36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a87b8707c522ed559fdb753e66d656d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4be20b6032c6b4bc2c02d0c0c5b3b36
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_472e3809386a3fa76ad957f16cfed20f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52bd55d59c4a743282e653701aa77158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.074566841125488]], [[5.675384044647217]], [[4.589181423187256]], [[5.357204914093018]], [[4.686102867126465]], [[5.177704811096191]], [[4.444948673248291]], [[4.822673320770264]], [[4.742612838745117]], [[4.711595058441162]], [[4.70556640625]], [[4.762353420257568]], [[5.579042434692383]], [[4.624544143676758]], [[5.49599027633667]], [[4.738476753234863]], [[5.141264915466309]], [[4.434568405151367]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_a87b8707c522ed559fdb753e66d656d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4be20b6032c6b4bc2c02d0c0c5b3b36
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f0035469aa8e3b2f0e59695eb2d443d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52a71d17b4b558771bb6a962af7e0051(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0035469aa8e3b2f0e59695eb2d443d
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65906b5bbd3456ff55e2c9fba3357bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d19887ab5405af22a88a3bc88352b401(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6a8716162cf0ee502dcb01428ee5481
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58b4d594936faba7c593dedca3816a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6d6ca038f216b1b1d6a25cf96ec5b77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6d6ca038f216b1b1d6a25cf96ec5b77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58b4d594936faba7c593dedca3816a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6d6ca038f216b1b1d6a25cf96ec5b77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6d6ca038f216b1b1d6a25cf96ec5b77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba863e743f26cbb51c28f3ff652ca504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_766d5ba22a8eab080dbec15aef876d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_766d5ba22a8eab080dbec15aef876d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81de974dcc7c4075e4abbaa649dd9260(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2caf30431daf360c62a28671c6479e01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2caf30431daf360c62a28671c6479e01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00259ae5dc877970f6638794e5a55582(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_405fcdcd2f0808739d06431a4de5e7d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_405fcdcd2f0808739d06431a4de5e7d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00259ae5dc877970f6638794e5a55582(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_405fcdcd2f0808739d06431a4de5e7d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_405fcdcd2f0808739d06431a4de5e7d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48a0a92adb86c53934a42e8166a21370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61578fb102d17ed124ca15c36fcfc1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61578fb102d17ed124ca15c36fcfc1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be09fcd93432e35b4d81b8afcf3e6091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0296dbff807317b05feceb32897d0c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0296dbff807317b05feceb32897d0c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1cbe8ace61bba897679e77a77994e48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_189690d114989cf31ecde125007ca5bc
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66e4dc4e11ccd2b5333e188cc6477397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e97f41c56433b71b4a055491d5f0160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_857c21fa392c6921805c4f02babeb3bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90cdecf849a3962f85eeef505def9afd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_364025fe188e47b3778d0580e34cc341(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7726372b1e960f332a8b37f81dc4b6bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e01901587f7c1f9983f1dce5c54611dc
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c28cda091fe8fba6c9bc97e3dd6c576b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6e63206b2cd2a48768260c405b1ebb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6e63206b2cd2a48768260c405b1ebb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ea3cda1b3d4b95809ed9460cae4d9e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ea3cda1b3d4b95809ed9460cae4d9e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63aae629205b2120bba7ddcf984a5dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63aae629205b2120bba7ddcf984a5dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63aae629205b2120bba7ddcf984a5dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6194dd0d7ed35201707f2561477220cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b486acca935f57835321b682ee33d72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6194dd0d7ed35201707f2561477220cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b486acca935f57835321b682ee33d72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6194dd0d7ed35201707f2561477220cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b486acca935f57835321b682ee33d72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6194dd0d7ed35201707f2561477220cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93c5141e0bc2614bfa7b921fae9dcc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6194dd0d7ed35201707f2561477220cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93c5141e0bc2614bfa7b921fae9dcc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6194dd0d7ed35201707f2561477220cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d93c5141e0bc2614bfa7b921fae9dcc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6194dd0d7ed35201707f2561477220cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d17d660db30157a39685dc9a7fac5ca1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96627e848f351a47a046bddf6dfcc1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d17d660db30157a39685dc9a7fac5ca1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96627e848f351a47a046bddf6dfcc1d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d17d660db30157a39685dc9a7fac5ca1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e261ddc362abc627d721e845fa7dd6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d8f579a6d0afdbea80aa934e11831a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6194dd0d7ed35201707f2561477220cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b54579c5ed976fc2268e1894d2c652d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0a632e3f78e91988ba778ba210437a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdff421a33cffd7ceaa380f6eed2fb8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccd59e01f780770640a731501f8e0651(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_626eade4c89992e5d0a59e328d6180aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f69dd133491a96cc5c8ddfe90eefd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c3a0fcc82621da1f682a313f10e0a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0905afa6c95cdcb4f7ad61333272620e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33ac0f0573f4de512ec370ae553528c
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5cfe4bead47149f591af6ba96da4e84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e01901587f7c1f9983f1dce5c54611dc
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af44ce564c2e1837a6c5c156e6031019(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a28531defd445c6b8835a3a1209c212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.495512008666992]], [[5.432677745819092]], [[4.943984508514404]], [[3.9269022941589355]], [[4.808197498321533]], [[5.317215442657471]], [[4.479256629943848]], [[5.208613395690918]], [[4.893153190612793]], [[4.050814151763916]], [[4.705148696899414]], [[5.471085071563721]], [[4.391146659851074]], [[4.680963516235352]], [[4.875914573669434]], [[4.115333557128906]], [[4.703266620635986]], [[4.208454132080078]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_275931edcb33e42ae3bd2cc41d482b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_595b65274641f25d6c1b5721129fc0d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dba17fc8a3d4e07e1f074d3039d89933(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccd100e22eaee704c29d5781a9dcf51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dba17fc8a3d4e07e1f074d3039d89933
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3107c5f02a00b940cbe6cf59ef1f8efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ae2d11ec8c5ef49477f1279b02de7d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6610159873962402]], [[4.382030487060547]], [[4.38055944442749]], [[4.22531270980835]], [[4.511491298675537]], [[4.1695027351379395]], [[3.865060806274414]], [[4.202451229095459]], [[4.938055515289307]], [[3.9300243854522705]], [[4.440571308135986]], [[3.968632459640503]], [[4.592230319976807]], [[3.6639225482940674]], [[4.616931915283203]], [[4.238045692443848]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_ba501ed01326d3aa6ee9b2638455add5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0035469aa8e3b2f0e59695eb2d443d
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f302e13d60873e0011cda37a731579(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.4651007652282715]], [[4.632149696350098]], [[4.672952651977539]], [[4.845834732055664]], [[4.880059242248535]], [[4.666262626647949]], [[4.770022392272949]], [[4.779326438903809]], [[5.001338958740234]], [[4.0377397537231445]], [[4.39913272857666]], [[4.714983940124512]], [[4.946009635925293]], [[4.713995456695557]], [[5.021440505981445]], [[4.741600513458252]], [[4.7320637702941895]], [[4.2691755294799805]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_0e6bbe86c698d3fe1608dbe889a3c53e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81236129c333dfe7ae73bbcbb0979cbf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.449842095375061]], [[1.7916018962860107]], [[1.7100330591201782]], [[1.0517908334732056]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_813e92b4b2c53228e8b8fc32cbe5bf22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6a8716162cf0ee502dcb01428ee5481
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0fe16e03bcc25cab99002506d00ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e825c4ee105986ee0867b4142aea0eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e825c4ee105986ee0867b4142aea0eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0fe16e03bcc25cab99002506d00ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e825c4ee105986ee0867b4142aea0eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e825c4ee105986ee0867b4142aea0eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a8b90ce1fe62d82748f3830c1c9fe78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88738c2c69098122f83be7ac0083fe81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88738c2c69098122f83be7ac0083fe81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01ad584cf45a74f64ae5a2f6a9e88a43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83b1cf996563ada24bd80f4a28dff5e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83b1cf996563ada24bd80f4a28dff5e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29414a23f1e8844d3ecf76672f89ad15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cd1153d39af14502a341f55e78131d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cd1153d39af14502a341f55e78131d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29414a23f1e8844d3ecf76672f89ad15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cd1153d39af14502a341f55e78131d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cd1153d39af14502a341f55e78131d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27953263c071c839f70b38706fc9cb65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd7096298ff18a0ba04fcbe8203732d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd7096298ff18a0ba04fcbe8203732d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56a16131cbafa5f8221d7f24d81b30a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_197f93a6256adee29789cd97bfc22695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_197f93a6256adee29789cd97bfc22695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98e220b346ff5b42d24dc32e543206ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_189690d114989cf31ecde125007ca5bc
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275931edcb33e42ae3bd2cc41d482b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_595b65274641f25d6c1b5721129fc0d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f3b48a5cf3b3d8362be7788a53b147a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79f904f616c9ebf47192fc807364cd83
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8945b1936d54b0700bb5042aa7bda998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33ac0f0573f4de512ec370ae553528c
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fe666a1201ee09dd6529861882eca4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b800aec2b225a238a1d1dac9bb71cd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe666a1201ee09dd6529861882eca4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ecc261b4deb38a3bbc195cf93c69a8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fe3b59475b5c4bf5c90842f4a790564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ecc261b4deb38a3bbc195cf93c69a8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90cdecf849a3962f85eeef505def9afd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275931edcb33e42ae3bd2cc41d482b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_595b65274641f25d6c1b5721129fc0d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a87b8707c522ed559fdb753e66d656d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4be20b6032c6b4bc2c02d0c0c5b3b36
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_091538d77ca5cac269cfc6395fab4045(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.5449934005737305]], [[5.682139873504639]], [[5.67764949798584]], [[5.4259419441223145]], [[6.047888278961182]], [[5.0137834548950195]], [[5.673959255218506]], [[5.87434720993042]], [[6.007184028625488]], [[5.486945629119873]], [[5.58497428894043]], [[5.264822006225586]], [[5.091436862945557]], [[6.080480098724365]], [[5.848564624786377]], [[5.892393112182617]], [[6.222475051879883]], [[5.960496425628662]], [[5.735707759857178]], [[6.294724464416504]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_356949426619f7638644f5b4ad10aad1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa362324846d7770124f03ef61e2598b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_356949426619f7638644f5b4ad10aad1
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1dd532fa575b929ebb34e47477432042(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27ce83676ffb6c30cbaf6578496cbc56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.1557114124298096]], [[2.9917681217193604]], [[3.779399871826172]], [[3.1832642555236816]], [[3.430224657058716]], [[3.3582873344421387]], [[2.883920669555664]], [[3.1816792488098145]], [[3.256134510040283]], [[3.319467306137085]], [[3.4573159217834473]], [[3.527083396911621]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_eb8d904ba4e4b7ebf01e1b7da584ad3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.775596618652344]], [[4.627426624298096]], [[5.101263999938965]], [[5.359864711761475]], [[5.436671257019043]], [[6.06939697265625]], [[5.455032825469971]], [[5.321661949157715]], [[5.425404071807861]], [[5.160300254821777]], [[5.353097438812256]], [[5.381292819976807]], [[4.7527241706848145]], [[5.453306198120117]], [[5.433315277099609]], [[5.309089660644531]], [[4.748722076416016]], [[5.337813377380371]], [[5.172105312347412]], [[5.770773410797119]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_4e72d2cd33b12576c4fc30b102001e76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a79f19dffaf1a401b1a360fa95eb71
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.358640193939209]], [[2.7018392086029053]], [[2.984464168548584]], [[2.702528953552246]], [[3.595785140991211]], [[2.438364267349243]], [[2.806942939758301]], [[3.076977252960205]], [[2.4327540397644043]], [[3.409235954284668]], [[2.8855111598968506]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fe3b59475b5c4bf5c90842f4a790564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ecc261b4deb38a3bbc195cf93c69a8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_094dbedb898587534c879e75e359d033(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa5522cc0fff55438ea3c29c97b2341b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8841710a027f5e25431f35fe689040ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa5522cc0fff55438ea3c29c97b2341b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.03779935836792]], [[3.275832176208496]], [[2.847233772277832]], [[3.191497802734375]], [[3.2837717533111572]], [[3.0297393798828125]], [[3.781230926513672]], [[3.339047908782959]], [[3.7403409481048584]], [[3.3798489570617676]], [[4.031096935272217]], [[3.373056173324585]], [[3.172370195388794]], [[3.249523162841797]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    
    class PrimitiveOp_fac7e4dfbb255c3a8dbf0c06d68a5027(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_456552b1550b21a6d465daa831b78da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fac7e4dfbb255c3a8dbf0c06d68a5027
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9c08f695dcae0c302451c1485c63a54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d36c42214c58fc9dac4eed51d6d02ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.5319061279296875]], [[5.019760608673096]], [[5.33010196685791]], [[5.282497882843018]], [[5.046922206878662]], [[5.342513561248779]], [[5.555017471313477]], [[5.292306900024414]], [[4.68528938293457]], [[4.734997272491455]], [[5.910179615020752]], [[4.748666286468506]], [[5.971632957458496]], [[5.610820770263672]], [[4.633826732635498]], [[5.679800510406494]], [[6.121128082275391]], [[5.734371662139893]], [[4.891031265258789]], [[5.260383129119873]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_6e23acaf4a62c3635606ec136ef148bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23acaf4a62c3635606ec136ef148bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23acaf4a62c3635606ec136ef148bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23acaf4a62c3635606ec136ef148bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d06a79e0a9078d999d99f26769322018(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6fd6a73983722e9268fe5c19bc91114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34974.4765625]], [[39784.2109375]], [[28418.41796875]], [[34156.3203125]], [[35816.92578125]], [[27483.953125]]], [[[34581.046875]], [[39333.0859375]], [[28099.552734375]], [[33769.2890625]], [[35419.4296875]], [[27170.34765625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_727725646be9e9351cdadb5e6ce7dd65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[39317.140625]], [[33706.33203125]], [[41887.5703125]], [[37642.73828125]], [[40813.0546875]], [[41225.6953125]]], [[[40262.890625]], [[34524.9296875]], [[42899.82421875]], [[38544.11328125]], [[41793.37109375]], [[42220.3203125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_886c9ef0784f980d0739b69f93b5da8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[39998.03125]], [[36421.7578125]], [[39325.65625]], [[47435.87109375]], [[31697.357421875]], [[40192.0703125]]], [[[41765.7421875]], [[38032.9765625]], [[41064.15234375]], [[49526.53515625]], [[33105.390625]], [[41965.94140625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_f00e881b9d6c8c6cab789219a9473ced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41410.9140625]], [[47603.7734375]], [[45140.16796875]], [[48028.58203125]], [[42334.01953125]], [[38568.15234375]]], [[[43256.73828125]], [[49733.0703125]], [[47160.5]], [[50171.8359375]], [[44224.015625]], [[40284.5]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef8550292e3da9a66aac4aaac3198735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b6f3ee13f03eb50a21f1a0136963d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcfd5ef1d4a2be9abedcdc12c66c976a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a87b8707c522ed559fdb753e66d656d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4be20b6032c6b4bc2c02d0c0c5b3b36
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf3e74628184a80f2df0207ce05ce8a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.444602012634277]], [[7.261509418487549]], [[8.290815353393555]], [[6.83220100402832]], [[7.594128131866455]], [[7.976133346557617]], [[7.4478888511657715]], [[6.68143367767334]], [[8.074603080749512]], [[7.364259719848633]], [[7.592048645019531]], [[7.421213626861572]], [[7.4837775230407715]], [[7.403409004211426]], [[8.940410614013672]], [[7.858551025390625]], [[6.976071834564209]], [[7.378269195556641]], [[7.4309563636779785]], [[7.936702251434326]], [[7.602663993835449]], [[7.9230146408081055]], [[7.234231472015381]], [[7.327571868896484]], [[7.961794376373291]], [[8.160552978515625]], [[8.463578224182129]], [[8.798234939575195]], [[8.465989112854004]], [[7.31309700012207]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_2b2bc26727eae20bbffc7919be1ded41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.720489501953125]], [[7.950204849243164]], [[8.676801681518555]], [[8.149105072021484]], [[7.102902412414551]], [[7.874119758605957]], [[8.240650177001953]], [[7.847740173339844]], [[7.982648849487305]], [[7.491360664367676]], [[8.735313415527344]], [[7.510238170623779]], [[8.066055297851562]], [[8.007878303527832]], [[8.85459041595459]], [[8.496610641479492]], [[8.094125747680664]], [[7.891778945922852]], [[8.22463607788086]], [[7.96538782119751]], [[7.587237358093262]], [[8.23629379272461]], [[8.250158309936523]], [[8.19361400604248]], [[8.194456100463867]], [[9.199542045593262]], [[7.542173385620117]], [[8.364714622497559]], [[7.895327091217041]], [[8.275627136230469]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_093ed3cdc8f4d1b21619a0d4aa800009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c977f2481f055abfce37bdb2deb32e49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.00590991973877]], [[7.415954113006592]], [[7.876989841461182]], [[6.942861080169678]], [[8.300236701965332]], [[7.491955757141113]], [[7.546222686767578]], [[7.608494758605957]], [[7.794553279876709]], [[7.298939228057861]], [[8.410886764526367]], [[7.41102409362793]], [[8.027511596679688]], [[7.821455478668213]], [[8.39918041229248]], [[8.598889350891113]], [[8.43173885345459]], [[8.188957214355469]], [[8.038849830627441]], [[7.986676216125488]], [[7.033593654632568]], [[7.4206156730651855]], [[8.053648948669434]], [[7.339644908905029]], [[7.243391513824463]], [[8.141550064086914]], [[7.59982967376709]], [[8.755999565124512]], [[8.071479797363281]], [[7.06417179107666]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_1c46bfaadeba0b3b94a4a8e8d0f166b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69484f284a4f8758dc40c42fb1ee6f7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c46bfaadeba0b3b94a4a8e8d0f166b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275931edcb33e42ae3bd2cc41d482b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_595b65274641f25d6c1b5721129fc0d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52a2483c0df7c0693c074ee1d94ac58b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.14487075805664]], [[8.147989273071289]], [[8.095406532287598]], [[7.036656856536865]], [[7.791210174560547]], [[7.92747688293457]], [[7.291300296783447]], [[7.665894031524658]], [[8.080425262451172]], [[7.467463970184326]], [[7.3757219314575195]], [[7.0921525955200195]], [[6.8974761962890625]], [[7.289829730987549]], [[7.877655982971191]], [[7.310974597930908]], [[7.013061046600342]], [[6.738163948059082]], [[7.917231559753418]], [[6.857556343078613]], [[8.298291206359863]], [[7.9575042724609375]], [[7.84797477722168]], [[8.23776912689209]], [[7.439065933227539]], [[8.512922286987305]], [[7.981821060180664]], [[7.359304428100586]], [[6.39490270614624]], [[7.269107818603516]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_c961f5254798bd9b662a1aa85ff6a7b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0138115882873535]], [[3.2304744720458984]], [[3.1001744270324707]], [[3.620997667312622]], [[3.425168514251709]], [[3.1345767974853516]], [[3.0422685146331787]], [[3.525670289993286]], [[3.386475086212158]], [[3.072099447250366]], [[3.4267210960388184]], [[2.8959836959838867]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_5070ce50072b726ab8f6293c946e1d29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.5646629333496094]], [[3.4088218212127686]], [[4.021111011505127]], [[2.847135305404663]], [[3.5908870697021484]], [[2.934852361679077]], [[3.508193016052246]], [[3.6472582817077637]], [[3.661288261413574]], [[3.0511858463287354]], [[3.5024564266204834]], [[3.0943455696105957]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_63e1a9cd73fb715e43ba40b160145ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.729383945465088]], [[7.137293338775635]], [[6.083255290985107]], [[6.352455139160156]], [[7.294790267944336]], [[7.118271827697754]], [[7.140522003173828]], [[7.952057838439941]], [[7.3802666664123535]], [[7.0913896560668945]], [[6.739432334899902]], [[7.589100360870361]], [[7.090548515319824]], [[6.952294826507568]], [[7.234918594360352]], [[7.350093364715576]], [[6.242755889892578]], [[6.655361175537109]], [[5.618881702423096]], [[7.279906272888184]], [[6.899659633636475]], [[7.120492935180664]], [[7.67257022857666]], [[7.347502708435059]], [[6.531886100769043]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    
    class PrimitiveOp_9455a0f9854f09439f255fbb68057e11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f2847c42771ec004ad1c8211ced5067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9455a0f9854f09439f255fbb68057e11
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d149f83e45bf1ef4bcdedb25a241c7b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89c1fc6df53dc054a2e4e9c362f6d6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d149f83e45bf1ef4bcdedb25a241c7b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a0a768dec3365501afcf2d0ad226468(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7ab8516e2fee0ef44ba229dc9a71d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0035469aa8e3b2f0e59695eb2d443d
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b392c2c335778c52fd0ba180e0e4339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_513dceeba098c4c9ba4592ead1b86e8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.62026834487915]], [[4.912600517272949]], [[4.501530170440674]], [[5.113858699798584]], [[5.229439735412598]], [[4.599358081817627]], [[5.316549777984619]], [[4.630554676055908]], [[4.967809200286865]], [[4.93235969543457]], [[5.05343770980835]], [[5.221227169036865]], [[4.371150493621826]], [[5.340354919433594]], [[5.351968288421631]], [[5.038957118988037]], [[4.813811779022217]], [[5.55605411529541]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_ad5329daa177ec911bcf0d0a3e17ef26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef2f7cd529c9f9bae202ad62a9bd4852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad5329daa177ec911bcf0d0a3e17ef26
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c8f86f432645babc0a68a941e70343e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3406821489334106]], [[1.40310800075531]], [[1.3875422477722168]], [[1.3541191816329956]], [[1.5349763631820679]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    
    class PrimitiveOp_b494d1026b11772bb7409431868099ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b0ab310ddba0dcb00ad2b581937fb05f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.891082525253296]], [[2.8003101348876953]], [[3.0362467765808105]], [[3.3644373416900635]], [[2.7299394607543945]], [[3.5558624267578125]], [[3.410123109817505]], [[3.1943938732147217]], [[3.5989999771118164]], [[2.9150657653808594]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    
    class PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e95413a8c45003faab305175bbe4179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.823153972625732]], [[5.129012584686279]], [[5.525485515594482]], [[4.958107948303223]], [[5.484551906585693]], [[4.751092910766602]], [[5.324677467346191]], [[6.043829917907715]], [[5.8095316886901855]], [[5.279562473297119]], [[5.300936222076416]], [[6.204381465911865]], [[5.513743877410889]], [[5.982616424560547]], [[5.167477607727051]], [[5.536433219909668]], [[4.966822147369385]], [[5.783491134643555]], [[4.380859851837158]], [[5.523407459259033]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69484f284a4f8758dc40c42fb1ee6f7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c46bfaadeba0b3b94a4a8e8d0f166b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b440e757310cef06249aecb8b8ba3a96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fe84f2534f9c690d4103c62c9fb6e84
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_472f90195d81e0ba99d29b026a6f2719(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11a218c469172afe6b54a19cb7f70088
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_663776b2cfae4b9439171c0b1b65da52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.116128921508789]], [[6.4880290031433105]], [[6.603219032287598]], [[5.883945465087891]], [[5.97711706161499]], [[6.399841785430908]], [[6.568505764007568]], [[7.100306987762451]], [[6.596995830535889]], [[6.940871238708496]], [[6.557913780212402]], [[6.544834613800049]], [[5.847147464752197]], [[7.082651138305664]], [[6.433468341827393]], [[6.7771406173706055]], [[6.6440534591674805]], [[6.463765621185303]], [[6.267218112945557]], [[6.189576625823975]], [[7.43379020690918]], [[6.055135250091553]], [[6.613389492034912]], [[6.876086711883545]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_761e0693c7faa426ce89a24460e11a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abc33e1d9d2d254bd23f9ac46c3df778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0780367851257324]], [[2.8307974338531494]], [[2.5395758152008057]], [[2.5772764682769775]], [[3.0108275413513184]], [[3.0733754634857178]], [[2.7212250232696533]], [[3.425830125808716]], [[3.5829856395721436]], [[2.8356549739837646]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_875ef62cd1ae41ca8f2c7ce033b45851(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09f8e84314e8e84c358d3ef709460020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_97ed7a90317f3f5f238086f72d9f7a6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8abf95a695fe6cf68198d7b278d0a23f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97ed7a90317f3f5f238086f72d9f7a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b870f7f6beef95d4ad6d3418f26916fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0035469aa8e3b2f0e59695eb2d443d
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_418c68ec84c0ba68b6c9c2c5010e0611(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.2628655433654785]], [[5.365011692047119]], [[5.038057804107666]], [[5.306423187255859]], [[5.208133697509766]], [[4.970277786254883]], [[5.335209369659424]], [[5.289772033691406]], [[4.597698211669922]], [[4.871645450592041]], [[4.971390724182129]], [[4.5013556480407715]], [[5.485369682312012]], [[4.997920036315918]], [[5.272904872894287]], [[4.238986968994141]], [[5.5695109367370605]], [[4.83064079284668]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_831862bc6d25ea81d4a94454e1a13a3c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c76575860ada744a3d5ae4bf4e5b951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_831862bc6d25ea81d4a94454e1a13a3c
        def get_inputs(self):
            return [
                paddle.to_tensor([[8.870004653930664, 9.070971488952637, 8.6998291015625, 8.771418571472168, 8.77302074432373, 8.278258323669434, 9.320741653442383, 8.51421070098877, 8.791091918945312, 8.457852363586426, 9.51916790008545, 8.142207145690918, 8.19686508178711, 8.17872428894043, 8.666762351989746, 8.191591262817383, 9.497175216674805, 8.180706024169922, 8.71663761138916, 9.759553909301758, 8.672922134399414, 8.273152351379395, 8.350447654724121, 9.068215370178223, 8.876349449157715, 8.625724792480469, 9.934979438781738, 8.560579299926758, 8.065917015075684, 9.131192207336426]], dtype='float32').reshape([1, 30]),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b800aec2b225a238a1d1dac9bb71cd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe666a1201ee09dd6529861882eca4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cab694990cd105f79fe2acf8ee63c25b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.780267715454102]], [[8.131998062133789]], [[8.939107894897461]], [[8.882832527160645]], [[9.453350067138672]], [[8.85619068145752]], [[8.147751808166504]], [[8.396537780761719]], [[8.775067329406738]], [[8.127144813537598]], [[8.213729858398438]], [[8.807175636291504]], [[8.307685852050781]], [[9.488317489624023]], [[7.320240497589111]], [[8.850549697875977]], [[8.130460739135742]], [[8.228346824645996]], [[8.956204414367676]], [[8.420777320861816]], [[8.549696922302246]], [[7.867123126983643]], [[8.182889938354492]], [[8.662212371826172]], [[7.8356781005859375]], [[8.064471244812012]], [[7.9256134033203125]], [[7.720449447631836]], [[9.509540557861328]], [[8.814489364624023]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_6952dc8fe655d0e81fe5d88e3092114a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa6a208551763b029a4175fcd015eae
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.213860273361206]], [[0.7169217467308044]], [[1.3732637166976929]], [[0.9356551766395569]], [[0.9903314113616943]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_48609448a7ffe7cbb2d294fe853ad1a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.469463348388672]], [[3.039804458618164]], [[3.2300972938537598]], [[2.7001450061798096]], [[2.6241466999053955]], [[2.543884515762329]], [[2.668975591659546]], [[1.845589518547058]], [[2.947535514831543]], [[2.841306686401367]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_10bc90c395809e580d84b9f3f19da6bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.91616153717041]], [[4.971972942352295]], [[4.818826675415039]], [[4.943068027496338]], [[4.595055103302002]], [[5.193109035491943]], [[4.6342620849609375]], [[4.798951625823975]], [[5.12747859954834]], [[4.536078929901123]], [[3.727442502975464]], [[4.675539016723633]], [[5.137940406799316]], [[5.504049777984619]], [[4.462343692779541]], [[4.524420738220215]], [[5.013905048370361]], [[5.048361778259277]], [[4.960768222808838]], [[5.547129154205322]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f4b5e5ec5f1b6ea6f95e18418f3a854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.84181809425354]], [[3.919459342956543]], [[3.943265914916992]], [[3.615178108215332]], [[4.343533515930176]], [[4.357351303100586]], [[3.5953638553619385]], [[4.600090503692627]], [[3.89821195602417]], [[4.208963871002197]], [[4.014341831207275]], [[4.141739368438721]], [[4.350880146026611]], [[4.032442569732666]], [[4.039931774139404]], [[4.096301555633545]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_7e97f41c56433b71b4a055491d5f0160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_857c21fa392c6921805c4f02babeb3bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ca286b9369ea606db657d731e5c916b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e60c4c5076019ac05a0bcb3114dcf82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ca286b9369ea606db657d731e5c916b
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b9dc3350b2d825cc8605162d18ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e708aa363d68fd907ffd6a2d3a799465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5b4b87921f40ac20bb8bfa4ce6a5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209d5cb520aef21b8bb33ef0a9afc3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2488790284adec886cc9e8a3597c90a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa362324846d7770124f03ef61e2598b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_356949426619f7638644f5b4ad10aad1
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_456552b1550b21a6d465daa831b78da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fac7e4dfbb255c3a8dbf0c06d68a5027
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56762715b1f4b5ab80a9021f93a9bf9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa5522cc0fff55438ea3c29c97b2341b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.8748505115509033]], [[3.848036527633667]], [[3.7055251598358154]], [[3.35248064994812]], [[2.893239974975586]], [[4.06261682510376]], [[3.7211146354675293]], [[4.051092624664307]], [[3.9451940059661865]], [[4.032584190368652]], [[4.147378921508789]], [[2.9157514572143555]], [[3.8111705780029297]], [[3.7591934204101562]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_f4993a78fbb335307cce5eb98f371d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.051861763000488]], [[5.754777431488037]], [[4.818517684936523]], [[4.752799034118652]], [[4.6054182052612305]], [[5.425618648529053]], [[5.631992340087891]], [[5.630918979644775]], [[5.270631313323975]], [[5.155634880065918]], [[5.055809020996094]], [[4.907738208770752]], [[5.484491348266602]], [[4.884391784667969]], [[5.892811298370361]], [[5.109838962554932]], [[5.606997489929199]], [[5.1103386878967285]], [[5.412006378173828]], [[4.9560112953186035]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_0f0d01cf069a799d19d0f3027c8ba901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b440e757310cef06249aecb8b8ba3a96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fe84f2534f9c690d4103c62c9fb6e84
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f630c99a39e62d8a94cd96362cb07e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.228400707244873]], [[7.572210311889648]], [[7.720287799835205]], [[7.640965461730957]], [[6.785443305969238]], [[8.622623443603516]], [[8.482710838317871]], [[7.288590431213379]], [[7.2805681228637695]], [[8.149312973022461]], [[7.05112361907959]], [[8.065946578979492]], [[7.8982014656066895]], [[8.205387115478516]], [[8.06624698638916]], [[7.016504287719727]], [[8.207212448120117]], [[6.216163158416748]], [[7.654832363128662]], [[7.563879013061523]], [[7.733938217163086]], [[7.262387275695801]], [[7.852588176727295]], [[7.543629169464111]], [[8.282045364379883]], [[7.994214057922363]], [[7.4732866287231445]], [[8.085190773010254]], [[8.281085968017578]], [[7.778512477874756]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e60c4c5076019ac05a0bcb3114dcf82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ca286b9369ea606db657d731e5c916b
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61cb8ddb4063eb5b1aa3d11506895938(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6a8716162cf0ee502dcb01428ee5481
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65e08b3023ff1d5ee7214566c951fc1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ccc8840c880e32dbee9bcff5095f132(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ccc8840c880e32dbee9bcff5095f132(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65e08b3023ff1d5ee7214566c951fc1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfafe3e10b4c08a5074c3cf26a0b021f
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ccc8840c880e32dbee9bcff5095f132(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ccc8840c880e32dbee9bcff5095f132(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f42937a18ff90a271e5a150951ed182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_368efbe7154253df1321096f7a598a3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_368efbe7154253df1321096f7a598a3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c1988ce22f320186228a1e2e11501b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd0d70152d81b501f2c6c38a2988d40d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a330616952e9258966565c060492adb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a330616952e9258966565c060492adb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc192a27a376203d9453b718f413122d
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feb7b9bc67800dd6026179e6979f6304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a79ea0e1dfbfbade673bd2f876ce5306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a79ea0e1dfbfbade673bd2f876ce5306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_feb7b9bc67800dd6026179e6979f6304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd55a18ade9826a559e5e0bc18dfe2d5
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a79ea0e1dfbfbade673bd2f876ce5306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a79ea0e1dfbfbade673bd2f876ce5306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_352ddd0cbe6aae4d76df9333a6d95eb3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18a78f97f72e865b6ebefefec215fab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbd73ce262b2b3d31ee2a3c004b06fd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbd73ce262b2b3d31ee2a3c004b06fd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c65a0a778e0562251f979df631fcb2d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cfaf42c790e87cbc1bddbc3192f84a2
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_375b6cf91a0fab761e8cb752b7e3cd29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_375b6cf91a0fab761e8cb752b7e3cd29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2aca0867bffe4b390ee1085b28d60be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_189690d114989cf31ecde125007ca5bc
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69484f284a4f8758dc40c42fb1ee6f7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c46bfaadeba0b3b94a4a8e8d0f166b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a815f6024d70cc918d69b9cfec4ced34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fe3b59475b5c4bf5c90842f4a790564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ecc261b4deb38a3bbc195cf93c69a8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10c70e68246fe67077b453d8afcac13d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.9753828048706055]], [[6.003129005432129]], [[7.127122402191162]], [[5.840324401855469]], [[6.891082763671875]], [[6.126930236816406]], [[7.248678684234619]], [[6.783142566680908]], [[6.2587175369262695]], [[5.727454662322998]], [[7.110597133636475]], [[6.120436668395996]], [[7.036701202392578]], [[6.22518253326416]], [[7.063243389129639]], [[6.6329193115234375]], [[6.64615535736084]], [[5.935981750488281]], [[6.5288004875183105]], [[7.519911289215088]], [[6.803238391876221]], [[6.365416526794434]], [[5.566114902496338]], [[7.818721294403076]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_da7ccb1e286210db935bbd01349ccdb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.455387115478516]], [[6.70116662979126]], [[6.931902885437012]], [[6.813783168792725]], [[7.750301361083984]], [[7.221365451812744]], [[6.216297626495361]], [[6.400246620178223]], [[7.3066630363464355]], [[7.40735387802124]], [[6.200910568237305]], [[7.3968400955200195]], [[6.916532039642334]], [[7.136009693145752]], [[7.048123359680176]], [[7.445611953735352]], [[6.951174259185791]], [[6.86937952041626]], [[7.565418243408203]], [[6.618014335632324]], [[7.1575422286987305]], [[6.011700630187988]], [[7.5044941902160645]], [[6.530730247497559]], [[6.473878383636475]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_c3fdbc30c8781a2459d502c4a300184d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.961212396621704]], [[2.8444416522979736]], [[3.0302486419677734]], [[3.4965062141418457]], [[4.392430305480957]], [[3.4449360370635986]], [[4.060918807983398]], [[2.69504451751709]], [[3.8072237968444824]], [[3.5579631328582764]], [[3.595301628112793]], [[3.4865787029266357]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b440e757310cef06249aecb8b8ba3a96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fe84f2534f9c690d4103c62c9fb6e84
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980a40d37020b4097f4ae2b1947f4d94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6b735795aa7b39bbb1819f8ebb0214a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a89bb02d678a7727526fdbd760a8fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b735795aa7b39bbb1819f8ebb0214a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e97f41c56433b71b4a055491d5f0160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_857c21fa392c6921805c4f02babeb3bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_805c70176e7152bf881ea5bf94b3c86f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92e84f25ee501abd31b97466fadc0da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58773cdb46fe7a0a61f51361742e1d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1242487d21e96fb08cccfba8a9ca13e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c023a308fc6e7227abb7b22f6d86f4cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1242487d21e96fb08cccfba8a9ca13e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ab07a91e0a695a4a6489bd75d30a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[706.5451049804688]], [[698.1089477539062]], [[733.5790405273438]], [[774.1317138671875]], [[702.3773803710938]], [[767.1760864257812]], [[719.3434448242188]], [[709.82080078125]], [[691.0403442382812]], [[681.920166015625]], [[771.4940795898438]], [[669.5833740234375]], [[718.261962890625]], [[709.01171875]], [[721.8114013671875]], [[735.890625]], [[706.6136474609375]], [[698.2988891601562]], [[718.6580200195312]], [[772.4765014648438]], [[658.5473022460938]], [[714.2171630859375]], [[651.621337890625]], [[754.21337890625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_0784751d63f6b9680ff7d27f1bed229e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[70.37177276611328]], [[75.7459487915039]], [[73.71358489990234]], [[74.7445068359375]], [[72.09110260009766]], [[75.07721710205078]], [[75.77609252929688]], [[74.45829010009766]], [[70.37089538574219]], [[78.90221405029297]], [[83.17375946044922]], [[70.34793090820312]], [[81.13215637207031]], [[73.42264556884766]], [[70.3205795288086]], [[73.20497131347656]], [[77.08792114257812]], [[72.10660552978516]], [[67.34578704833984]], [[75.58836364746094]], [[74.80601501464844]], [[75.42642211914062]], [[81.91889953613281]], [[68.68610382080078]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_8b50bb239944d5f006be8e5a925165ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45.29734420776367]], [[51.2456169128418]], [[43.81867218017578]], [[44.45344924926758]], [[44.131744384765625]], [[45.768131256103516]], [[46.679420471191406]], [[42.54667663574219]], [[38.03548812866211]], [[47.30936813354492]], [[46.49201202392578]], [[46.001251220703125]], [[39.304664611816406]], [[44.54432678222656]], [[44.52653121948242]], [[44.796451568603516]], [[49.56315231323242]], [[45.860416412353516]], [[42.550559997558594]], [[45.2556266784668]], [[46.87994384765625]], [[43.06817626953125]], [[48.176002502441406]], [[40.50491714477539]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f7f3db40c286924aa18bc8ab4d43a5a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[22.481063842773438]], [[22.24987030029297]], [[22.205080032348633]], [[23.617107391357422]], [[22.71398162841797]], [[22.377277374267578]], [[23.214012145996094]], [[21.080718994140625]], [[21.404701232910156]], [[20.12889289855957]], [[23.075468063354492]], [[21.405946731567383]], [[23.80116081237793]], [[23.009891510009766]], [[20.116384506225586]], [[22.51435661315918]], [[22.5460205078125]], [[23.645549774169922]], [[22.766191482543945]], [[21.108200073242188]], [[21.514690399169922]], [[21.464120864868164]], [[23.93003273010254]], [[21.577220916748047]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_672b693697bf69ca84197b380f56927d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33464.0625]], [[34128.3203125]], [[34373.05859375]], [[28562.703125]], [[32737.2890625]], [[36944.26953125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_9d76f4d4901a44a1e82ba954aa5e1ece(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[40044.80078125]], [[37509.421875]], [[43190.484375]], [[37027.70703125]], [[37379.66015625]], [[33082.0859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_8fdff045a2e7cb9a0743f749901ddf72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45157.8984375]], [[35525.29296875]], [[38019.31640625]], [[36812.3671875]], [[32309.341796875]], [[43327.96484375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_8e9c7bc37c8ff97f95e157beab68559c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[46890.375]], [[40730.390625]], [[41829.359375]], [[43960.64453125]], [[42763.0546875]], [[40935.3671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_7b8a417f78c6855a49c723f7105fdef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f2847c42771ec004ad1c8211ced5067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9455a0f9854f09439f255fbb68057e11
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3bf90131f37ce068e0de0edae491929(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2f55f17f5ceb9036367e6de807e3351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.723304271697998]], [[6.127195835113525]], [[5.915198802947998]], [[5.47707986831665]], [[5.763934135437012]], [[5.835768699645996]], [[6.205861568450928]], [[5.506523132324219]], [[5.576387405395508]], [[5.98674201965332]], [[5.432145595550537]], [[5.712860584259033]], [[5.408754825592041]], [[5.702744960784912]], [[6.343686103820801]], [[6.0185065269470215]], [[5.901707172393799]], [[5.767567157745361]], [[5.454860687255859]], [[5.4875054359436035]], [[6.7340288162231445]], [[6.339094161987305]], [[6.3797926902771]], [[5.385677337646484]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    
    class PrimitiveOp_11be419fc9f85858794f580ec076980e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74386d138e1af98b08abe37048d63ae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11be419fc9f85858794f580ec076980e
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f6c07cde8e4a585b54333a3725e4809c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8234cbcc80f497a05afec08ab311d3c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6c07cde8e4a585b54333a3725e4809c
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_400e4b7354296ae93df0d425392c3d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c08707d8b252c03051dfd25790e353
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28d14bba1eb0b314bc460ccd1aae6a38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f0304a8d2fafa0584ac2074dc2761b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d14bba1eb0b314bc460ccd1aae6a38
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3fa5482916cffc526445466b37baf168(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_677e3fea87571367a5dd801711c000fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa5482916cffc526445466b37baf168
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.626307010650635, 4.391082763671875, 4.806268215179443, 4.401785850524902, 4.369851112365723, 4.688978672027588, 4.786715984344482, 4.2709174156188965, 4.522144794464111, 4.13695764541626, 5.095004558563232, 4.644773006439209, 3.8188412189483643, 4.205009937286377, 4.518152236938477, 4.431371212005615, 4.864565849304199, 3.6953768730163574]], dtype='float32').reshape([1, 18]),
            ]


    
    class PrimitiveOp_3e722c534a85f6265bab20f6209cc641(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_329d6ea097edc6ad333cb5af4da1d476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e722c534a85f6265bab20f6209cc641
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.27309513092041, 5.833820343017578, 5.503009796142578, 5.4817891120910645, 5.425642013549805, 6.097849369049072, 5.734645843505859, 5.922000885009766, 5.353610038757324, 5.361513137817383, 4.979347229003906, 6.077544212341309, 5.27100133895874, 5.504151821136475, 5.766942977905273, 5.257235527038574, 5.357554912567139, 5.529504299163818, 5.592182159423828, 5.393463611602783, 5.493752479553223, 6.035822868347168, 5.610565185546875]], dtype='float32').reshape([1, 23]),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2fcd5bc69e135a5ccb232dadaaba4c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2abe3dbcba51863829022e5b9df706fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2fcd5bc69e135a5ccb232dadaaba4c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b2dab206b3f73ef8b301a8b85d2b5af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e46677790eac75526b611c12b2e5ce96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b2dab206b3f73ef8b301a8b85d2b5af
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6cc79560084fd34421102d994ca57852(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 20, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ae161ceb755fe1c7379296ae98c7eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cc79560084fd34421102d994ca57852
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ede169fdc9d6bf589626f21954553494(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ef1104bc488156bb5b7bbc9ff5fdfe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ede169fdc9d6bf589626f21954553494
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ef1104bc488156bb5b7bbc9ff5fdfe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ede169fdc9d6bf589626f21954553494
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 168, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 30, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c448af68b73ec77bc56a5c73d97776f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.365582466125488]], [[7.731196403503418]], [[7.964057445526123]], [[6.985565662384033]], [[7.5866594314575195]], [[8.946654319763184]], [[7.98096227645874]], [[8.842916488647461]], [[7.250166893005371]], [[7.330620765686035]], [[7.9806976318359375]], [[8.2101411819458]], [[8.502208709716797]], [[7.358593940734863]], [[7.772006988525391]], [[7.207477569580078]], [[7.093987464904785]], [[8.185877799987793]], [[8.008213996887207]], [[7.6435675621032715]], [[7.370347499847412]], [[7.112212181091309]], [[7.433649063110352]], [[7.9769439697265625]], [[7.491349697113037]], [[8.110529899597168]], [[7.971155166625977]], [[8.05787467956543]], [[7.60438346862793]], [[7.9871745109558105]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_fc8405c8dc00d11843d8deda02d87197(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_177aab49fa33a6f90da440e5529d03ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc8405c8dc00d11843d8deda02d87197
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 80, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b8a37c9b155a9391f481fe4452925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c7c8648423a056e26a40fac51a9ddcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 40, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaddf904f965a35f6e5f91d0d37a8be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1090a73b4e06d6d64bc3d751251766c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_023876bf322ea71ba7a345f24968e6eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bba8832ab934b048536037972244a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_023876bf322ea71ba7a345f24968e6eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c1cc2480b2e46faa2271589fdc286850(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 10, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5426abba745b1f11500ef581f7d423dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 5, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be645c148e6554bf16ee4e752f975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5426abba745b1f11500ef581f7d423dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9413cf7084393af9ec3e2798b157f06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.194438934326172]], [[8.588522911071777]], [[7.182609558105469]], [[7.648037433624268]], [[6.894355773925781]], [[6.827334880828857]], [[7.831111431121826]], [[7.7364959716796875]], [[8.895010948181152]], [[7.1844940185546875]], [[7.992562294006348]], [[7.785795211791992]], [[8.145564079284668]], [[8.470157623291016]], [[8.22587776184082]], [[7.272141933441162]], [[8.16726303100586]], [[8.357715606689453]], [[7.874889850616455]], [[7.3341898918151855]], [[7.798108100891113]], [[7.923760414123535]], [[8.171854972839355]], [[8.929399490356445]], [[6.554434299468994]], [[7.777000427246094]], [[8.502005577087402]], [[8.291655540466309]], [[8.365804672241211]], [[8.238239288330078]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_08c0a34c463f5accdbc2e9473cef8cf0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_414e9ca19a3c8b2dc36fb6e82a449b14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08c0a34c463f5accdbc2e9473cef8cf0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e30d4d09e7bcb5d7c9ff5854e58ff6a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5952028036117554]], [[1.1798752546310425]], [[1.684241533279419]], [[1.4642938375473022]], [[1.7384600639343262]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_04a5b2e68911fe3c0bb48c6168200666(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.7241156101226807]], [[2.818718194961548]], [[3.44044828414917]], [[3.0282323360443115]], [[2.9832065105438232]], [[2.824711799621582]], [[3.593562126159668]], [[3.125070095062256]], [[2.591668128967285]], [[3.1199605464935303]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    
    class PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cfd0e892baee8a9a0f56dcdacfeadb4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.417937755584717]], [[5.54685640335083]], [[6.385406494140625]], [[6.424187660217285]], [[6.714924335479736]], [[5.881267547607422]], [[5.836642742156982]], [[5.545405864715576]], [[6.1585845947265625]], [[6.060319900512695]], [[5.8800458908081055]], [[6.286401748657227]], [[6.463059902191162]], [[5.401740074157715]], [[5.785524368286133]], [[5.819664478302002]], [[6.0124735832214355]], [[6.33400297164917]], [[6.509230613708496]], [[6.19317626953125]], [[6.521552085876465]], [[6.344600677490234]], [[5.961433410644531]], [[6.324902534484863]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    
    class PrimitiveOp_a9ca567cbb9dcf0b1a2b337392ca3f2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1eb285cab3e0674c13dc32dd6b1ca8ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9ca567cbb9dcf0b1a2b337392ca3f2d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c20d892bd43816678cbde0e09c267bb1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6757ea9a9b63f49de33939ad09e6eb4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c20d892bd43816678cbde0e09c267bb1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6df34950da61687debb9fabf3f934692(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95ca198ba0eaeb437bd15fac9af28dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6df34950da61687debb9fabf3f934692
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 18, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cef4b860ed394579155fb9b662d10919(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.443090915679932]], [[4.749655723571777]], [[3.8140125274658203]], [[4.799515247344971]], [[4.431580066680908]], [[4.186607837677002]], [[4.143815994262695]], [[4.519565582275391]], [[4.756207466125488]], [[4.297165393829346]], [[3.9445648193359375]], [[4.05826473236084]], [[4.993553638458252]], [[4.120225429534912]], [[4.165502071380615]], [[4.1748857498168945]], [[3.941878080368042]], [[4.543561935424805]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34c7e3413219a4d36a9a6ccab8c60ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.8558502197265625]], [[6.249524116516113]], [[5.586520195007324]], [[5.772902965545654]], [[6.590846538543701]], [[6.154374122619629]], [[6.037872314453125]], [[5.574150085449219]], [[5.730854034423828]], [[5.262627124786377]], [[5.772827625274658]], [[6.190662384033203]], [[5.621458053588867]], [[5.770803451538086]], [[5.573040962219238]], [[5.884099960327148]], [[5.568533897399902]], [[5.506991863250732]], [[6.336510181427002]], [[5.539333820343018]], [[5.555763244628906]], [[6.508022308349609]], [[5.37600564956665]], [[5.509634971618652]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    
    class PrimitiveOp_edcc35934ec4117979abf3b17cee8c5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f848c558f8b62a6529eced125b2aac53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edcc35934ec4117979abf3b17cee8c5b
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fe38875aa6a776cea57fa50887a6489(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 28, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4cf2c2fc7fca564747c9ac3639a933af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fe38875aa6a776cea57fa50887a6489
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d9a5be699233ae66d2644ef9ba39603(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6591b39cfba0bec545e475776412286(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9a5be699233ae66d2644ef9ba39603
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9399208426475525]], [[0.7881795167922974]], [[0.7001737952232361]], [[1.489819884300232]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_f848c558f8b62a6529eced125b2aac53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edcc35934ec4117979abf3b17cee8c5b
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85b01dc5b87514b414a7a8aee3b35d0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3621a68275c8ae5b1aadff3f7b1e097d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85b01dc5b87514b414a7a8aee3b35d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.274381637573242]], [[3.215075969696045]], [[2.314345121383667]], [[2.7561116218566895]], [[3.2190792560577393]], [[3.261092185974121]], [[2.7221601009368896]], [[2.923419237136841]], [[3.007906436920166]], [[3.288865566253662]], [[3.1404919624328613]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a25c7b355f54cf118252117a9200253(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eb1f42c62ba63db1621a33954e58bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.629995346069336]], [[7.557254314422607]], [[8.062764167785645]], [[8.629494667053223]], [[7.266910552978516]], [[7.732415199279785]], [[7.328917980194092]], [[7.593316555023193]], [[7.870702743530273]], [[7.665899753570557]], [[8.408510208129883]], [[7.446771144866943]], [[8.450179100036621]], [[8.21670913696289]], [[8.057311058044434]], [[8.340054512023926]], [[8.768411636352539]], [[8.051828384399414]], [[8.499567985534668]], [[8.646636009216309]], [[7.833431720733643]], [[7.5960564613342285]], [[8.436894416809082]], [[8.195596694946289]], [[8.104879379272461]], [[7.728816032409668]], [[8.08309268951416]], [[7.9412055015563965]], [[8.761932373046875]], [[8.15397834777832]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cbe09b902245dff6f9978188b43477af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dc7f82313d6514cf3292c6b536832d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbe09b902245dff6f9978188b43477af
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23594abb7a87ed74d455a1805b056435(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 80, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc0ed0885d88ed55f7667eccbadb299c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23594abb7a87ed74d455a1805b056435
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62ec2d0c8d5b716033f298eb55184c66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.589501857757568]], [[3.9595940113067627]], [[4.664587020874023]], [[4.551024913787842]], [[4.109001636505127]], [[4.198220252990723]], [[4.432218551635742]], [[4.4817118644714355]], [[3.9227426052093506]], [[4.625372409820557]], [[3.8187661170959473]], [[4.21016788482666]], [[4.568661689758301]], [[4.094541072845459]], [[4.191646575927734]], [[4.210667133331299]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    
    class PrimitiveOp_125eb9938ce7b1eb71c444ba12c26b7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 14, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96cd7c91ef9ac75b2bb01d7e03f57348(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_125eb9938ce7b1eb71c444ba12c26b7e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a43e71f485c9d63fcd3251833255f79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 22, 33], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5eff16ddf435c6c6160ce807a07923c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a43e71f485c9d63fcd3251833255f79
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c93fcabd06e6af8be5bc2dace329b584(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 23, 35], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c60af22fb3160a5fbdc89808d87984d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93fcabd06e6af8be5bc2dace329b584
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a8ace7c78f3f496df9445e9493b9af8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 46, 70], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b4fbfad0f25408329482baf10b3aec4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a8ace7c78f3f496df9445e9493b9af8
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1df041cb455c0a3f83f0911a276d267d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_275390e0aba907edcbbf42bd5f5efd60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df041cb455c0a3f83f0911a276d267d
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5532d1565aef070205a5dc643dd8ac6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32e33dfe42ad4d5242a8d1f84376f4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5532d1565aef070205a5dc643dd8ac6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb21c8249f9f100cae27ce59fb5a4424(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.331165313720703]], [[6.932826519012451]], [[6.909614086151123]], [[7.851978302001953]], [[7.19744348526001]], [[7.731747627258301]], [[6.7408857345581055]], [[7.960355281829834]], [[6.684189796447754]], [[7.306125164031982]], [[6.787485122680664]], [[6.810930252075195]], [[6.970196723937988]], [[7.652226448059082]], [[7.5926666259765625]], [[7.016533374786377]], [[6.701245307922363]], [[7.672969818115234]], [[7.558865547180176]], [[7.018603324890137]], [[7.665589332580566]], [[7.1204938888549805]], [[6.956162452697754]], [[7.111342906951904]], [[7.7293500900268555]], [[7.124806880950928]], [[7.486032962799072]], [[7.5760722160339355]], [[7.356403350830078]], [[7.904211521148682]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_6f7bf0bc2afb7d4921ef4cd868b8bbea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30706f1944ebd6943d0de63e07ff03b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f7bf0bc2afb7d4921ef4cd868b8bbea
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3eab7a4665cb38068f08b87124ed65e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6d84af4fe1e2daadbfda76d384aab2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3eab7a4665cb38068f08b87124ed65e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 25, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6833158e5e5e2caf37d2152ecc6775b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.2826313972473145]], [[6.334243297576904]], [[6.943402290344238]], [[6.860344886779785]], [[7.050297737121582]], [[7.861888408660889]], [[6.671663284301758]], [[6.760298728942871]], [[7.774744510650635]], [[7.529026985168457]], [[8.103219032287598]], [[6.956748008728027]], [[7.525053024291992]], [[6.947916030883789]], [[6.68371057510376]], [[7.102306365966797]], [[6.944980144500732]], [[6.7735161781311035]], [[7.2324910163879395]], [[6.96896505355835]], [[7.399407386779785]], [[6.78544807434082]], [[8.230155944824219]], [[6.772273540496826]], [[7.684123516082764]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_244f4bdfa9406d54d7ca2a933b2aebad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 6, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e544e006a60a24e5075fa0659386c178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_244f4bdfa9406d54d7ca2a933b2aebad
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f85328aac18b5531513e7e776d667093(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_444ffcc16bcc6ef8b6478fc2c52a8fa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85328aac18b5531513e7e776d667093
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb702e2c14f93a767ca267caa9f13e2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[390, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef3767e5dd45a38f8f305ad12f605f1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb702e2c14f93a767ca267caa9f13e2e
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef3767e5dd45a38f8f305ad12f605f1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb702e2c14f93a767ca267caa9f13e2e
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c151052102c3491c412426e5ddc94da0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_447a102c8c794c72fb78f0466a2d82a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c151052102c3491c412426e5ddc94da0
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a9c36a1e5b6752923abfacd8ee2892d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 60, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d6784961e85cb5e80efafbadcc41c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a9c36a1e5b6752923abfacd8ee2892d
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_116a98b5d05d7c1c9cbe1eae5d562663(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.324487686157227]], [[5.197154998779297]], [[5.39374303817749]], [[4.502297401428223]], [[5.0500264167785645]], [[5.456243515014648]], [[4.709370136260986]], [[5.073709487915039]], [[5.223665714263916]], [[5.386110782623291]], [[4.320528507232666]], [[5.036959171295166]], [[5.214683532714844]], [[5.159992694854736]], [[5.648549556732178]], [[5.104668140411377]], [[5.089097499847412]], [[4.991738319396973]], [[4.9874982833862305]], [[4.739211559295654]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30706f1944ebd6943d0de63e07ff03b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f7bf0bc2afb7d4921ef4cd868b8bbea
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a22ccc03d15092ba855978baa7059f71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.766676902770996]], [[4.8825483322143555]], [[4.5546064376831055]], [[4.451656818389893]], [[4.954763889312744]], [[5.02230167388916]], [[4.508750915527344]], [[4.965527534484863]], [[4.956589698791504]], [[4.86350679397583]], [[4.453602313995361]], [[4.658884048461914]], [[4.835541725158691]], [[4.794821739196777]], [[4.3191351890563965]], [[4.349730968475342]], [[4.615575790405273]], [[5.2794270515441895]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_e842830000162c9057ef7e32de809e24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64effd5b1d34c0379e1e0b6b1cf0918d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e842830000162c9057ef7e32de809e24
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f0304a8d2fafa0584ac2074dc2761b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d14bba1eb0b314bc460ccd1aae6a38
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3d8e322defe8342121f42538f452d8a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc6598482c15f718aa6ed6e0055e0fed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d8e322defe8342121f42538f452d8a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7cb88b29b534217215292369f6e38209(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 109, 109], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_001418cf16644f519c650c68721b4eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cb88b29b534217215292369f6e38209
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1279ceafd50802130b7bb8a406277855(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 16, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4a36aaa3bc97aae653b75a4c3d4f038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1279ceafd50802130b7bb8a406277855
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_139a3f78bcd3e387d73cc68c0033c62a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 64, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5996156dfec2d6092482b2db286722c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_139a3f78bcd3e387d73cc68c0033c62a
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5996156dfec2d6092482b2db286722c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_139a3f78bcd3e387d73cc68c0033c62a
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4a36aaa3bc97aae653b75a4c3d4f038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1279ceafd50802130b7bb8a406277855
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5996156dfec2d6092482b2db286722c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_139a3f78bcd3e387d73cc68c0033c62a
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5996156dfec2d6092482b2db286722c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_139a3f78bcd3e387d73cc68c0033c62a
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a91b0571617b8474e3513ac18b80a103(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 32, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41362a376fb13ea3739ab9f2820593af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91b0571617b8474e3513ac18b80a103
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7acbdaf205fa93f08c2236a443ab7e25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 128, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41b291437e2b85376bda0d6b6449713f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acbdaf205fa93f08c2236a443ab7e25
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41b291437e2b85376bda0d6b6449713f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7acbdaf205fa93f08c2236a443ab7e25
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_20212e7e1d4a640673df97f0e69d046d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 32, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43e8e99a388b52bd46e26912b13f44b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20212e7e1d4a640673df97f0e69d046d
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_69185ba1e024567a292566c1ee5c990f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 128, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdc204de3546020661db112f144a6537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69185ba1e024567a292566c1ee5c990f
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdc204de3546020661db112f144a6537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69185ba1e024567a292566c1ee5c990f
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6883dc57dc26d5237c2829d6b48f84cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 48, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2896a89447a2319dea7620b8c6df56be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6883dc57dc26d5237c2829d6b48f84cf
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a3a895ef6a07139b8af29fd2fc3a9cbe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41d69e43a65c58708d09214c41f0b1d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3a895ef6a07139b8af29fd2fc3a9cbe
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41d69e43a65c58708d09214c41f0b1d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3a895ef6a07139b8af29fd2fc3a9cbe
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2896a89447a2319dea7620b8c6df56be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6883dc57dc26d5237c2829d6b48f84cf
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41d69e43a65c58708d09214c41f0b1d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3a895ef6a07139b8af29fd2fc3a9cbe
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41d69e43a65c58708d09214c41f0b1d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3a895ef6a07139b8af29fd2fc3a9cbe
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d411f05a575b9f381dff011060568caa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 64, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6095073ff4a0b4cea6e4898a759173ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d411f05a575b9f381dff011060568caa
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7de72597ea9894b1e49bed4350b0098(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 256, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_767c6e985c72ca210fc5e94a9a61efd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7de72597ea9894b1e49bed4350b0098
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_767c6e985c72ca210fc5e94a9a61efd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7de72597ea9894b1e49bed4350b0098
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4cf2830b42c91a13cfdfb92577e82343(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 64, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a374b892b51f560ece90432a28863c8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cf2830b42c91a13cfdfb92577e82343
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0ffb2543f95b4fd28479d9eaeea0215(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 256, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c3842e1cd9e59c73823471ea34e91457(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ffb2543f95b4fd28479d9eaeea0215
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3842e1cd9e59c73823471ea34e91457(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ffb2543f95b4fd28479d9eaeea0215
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_638cbb7cdd22049782b427d6ca2dafd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1000, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2373c0060be8e6d5ad6c2d45170ebbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_638cbb7cdd22049782b427d6ca2dafd1
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d6784961e85cb5e80efafbadcc41c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a9c36a1e5b6752923abfacd8ee2892d
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e9e351c08edd9292174f1691bf61456(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.547247409820557]], [[5.241771221160889]], [[4.631717205047607]], [[5.429015636444092]], [[4.983003616333008]], [[4.7412495613098145]], [[4.678460597991943]], [[5.05424165725708]], [[4.121486186981201]], [[4.833622932434082]], [[4.7639336585998535]], [[4.579180717468262]], [[4.328996181488037]], [[4.869755744934082]], [[5.027015686035156]], [[4.672824859619141]], [[4.301534652709961]], [[4.660597324371338]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_177aab49fa33a6f90da440e5529d03ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc8405c8dc00d11843d8deda02d87197
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c11b8c880f7b8f01448bb43f2a7299a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.95380163192749]], [[5.899272918701172]], [[6.4619035720825195]], [[6.993582725524902]], [[6.350670337677002]], [[6.611702919006348]], [[6.624584197998047]], [[6.251131534576416]], [[5.933366775512695]], [[5.4472479820251465]], [[6.299964904785156]], [[6.762087345123291]], [[6.120079040527344]], [[6.5732316970825195]], [[5.486151218414307]], [[6.62627649307251]], [[6.923728942871094]], [[6.560196399688721]], [[6.101138591766357]], [[6.05964994430542]], [[7.280458450317383]], [[6.875477313995361]], [[5.495212078094482]], [[6.6487884521484375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    
    class PrimitiveOp_4de82e4f2f43bbe0890bcd803851c2b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 11, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56aa0edcb2c86bdb34e4e8d7fed6bfb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4de82e4f2f43bbe0890bcd803851c2b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f35376f52ee66a2c0981b5345279581(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.828707695007324]], [[4.706035137176514]], [[4.936190605163574]], [[4.7714033126831055]], [[5.0137248039245605]], [[4.696411609649658]], [[4.654387950897217]], [[4.914733409881592]], [[4.297507286071777]], [[4.808157444000244]], [[4.754739284515381]], [[4.652151107788086]], [[4.769132614135742]], [[5.319541931152344]], [[5.285849094390869]], [[4.965315818786621]], [[5.107652187347412]], [[5.249327659606934]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_b9d9bcc1ddd3729d465db05074aabcec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d24da0d46548f30fe843556043ec2aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9d9bcc1ddd3729d465db05074aabcec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a04c3c749719846c2f2572d59f93d4ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 10, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94e4d108331c7edff6631422f7f6df63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a04c3c749719846c2f2572d59f93d4ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eda845c9d087f81035cc310aa5a45b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.074566841125488]], [[5.675384044647217]], [[4.589181423187256]], [[5.357204914093018]], [[4.686102867126465]], [[5.177704811096191]], [[4.444948673248291]], [[4.822673320770264]], [[4.742612838745117]], [[4.711595058441162]], [[4.70556640625]], [[4.762353420257568]], [[5.579042434692383]], [[4.624544143676758]], [[5.49599027633667]], [[4.738476753234863]], [[5.141264915466309]], [[4.434568405151367]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_4d24da0d46548f30fe843556043ec2aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9d9bcc1ddd3729d465db05074aabcec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9992bd2cf31647e733103cb1499ec527(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d335e842b9c9a8cd916db02214f830e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9992bd2cf31647e733103cb1499ec527
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a61aa1e9ab2e8f159fecd4d70566e807(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 12, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca4710912d35b8eaf6d08e350e3d220e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a61aa1e9ab2e8f159fecd4d70566e807
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5569cd496f811d8c3bdf0f3426e1b73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 96, 109, 109], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33c00deaf1bd7e7eaea2fd3c6bcdc9a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5569cd496f811d8c3bdf0f3426e1b73
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f09a3f6ecf039d49450ea5a5253aee2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 16, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38673d3e07fa913e6d7aa58cfb03d69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f09a3f6ecf039d49450ea5a5253aee2
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61bd37e312ca167203865588faca8795(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 64, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_93e620255d1a690e78294fe3d95dc278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61bd37e312ca167203865588faca8795
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93e620255d1a690e78294fe3d95dc278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61bd37e312ca167203865588faca8795
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38673d3e07fa913e6d7aa58cfb03d69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f09a3f6ecf039d49450ea5a5253aee2
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93e620255d1a690e78294fe3d95dc278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61bd37e312ca167203865588faca8795
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93e620255d1a690e78294fe3d95dc278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61bd37e312ca167203865588faca8795
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d208a82ec6e1a637d34f614fe656aa33(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c68c3a14d4b0353980e852edbb6b361f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d208a82ec6e1a637d34f614fe656aa33
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca2bde8f7c63b23ae70b16d773480b78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10e38c50c8557fdd712c057a72c07b0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca2bde8f7c63b23ae70b16d773480b78
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10e38c50c8557fdd712c057a72c07b0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca2bde8f7c63b23ae70b16d773480b78
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4528a43761873f008a334518d853eb06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70aced109dff1882c07d732ebaa1b309(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4528a43761873f008a334518d853eb06
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1dd466119ad0d183e62ff527676c6e90(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8da78b18fc6d86ffea3defa22f5d9a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd466119ad0d183e62ff527676c6e90
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8da78b18fc6d86ffea3defa22f5d9a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd466119ad0d183e62ff527676c6e90
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f6230eb609fe028376f4a5e01b8075af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 48, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92c529f8df96d2d608393ad1267708d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6230eb609fe028376f4a5e01b8075af
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fca54abe451c314ba902577ea4d7906c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 192, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd24cb37c436d8ed47088d26ef8eec32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fca54abe451c314ba902577ea4d7906c
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd24cb37c436d8ed47088d26ef8eec32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fca54abe451c314ba902577ea4d7906c
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c529f8df96d2d608393ad1267708d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6230eb609fe028376f4a5e01b8075af
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd24cb37c436d8ed47088d26ef8eec32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fca54abe451c314ba902577ea4d7906c
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd24cb37c436d8ed47088d26ef8eec32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fca54abe451c314ba902577ea4d7906c
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_de46f8f73d10a1a799423fc9bd62e21e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 64, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_786613c8a308c7abe181c4052d4b35ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de46f8f73d10a1a799423fc9bd62e21e
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_834c6c3514423408e0f0bd733ddb5f6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c5fbb2c0a3f82b3dce48cf510efd491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_834c6c3514423408e0f0bd733ddb5f6f
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c5fbb2c0a3f82b3dce48cf510efd491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_834c6c3514423408e0f0bd733ddb5f6f
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e0abbd4af6fcd6e8aab4eb0845627d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 64, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b31701397006a6a8c756527fdf755ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e0abbd4af6fcd6e8aab4eb0845627d6
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6ccd82dfe10457bdd15c82213130425(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87ab296323ad66bd50f49dc9622022ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6ccd82dfe10457bdd15c82213130425
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87ab296323ad66bd50f49dc9622022ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6ccd82dfe10457bdd15c82213130425
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c67ffd61c0b2afe55377f53f3b994656(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1000, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a94646b09f3d4a163763ecae2305b169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c67ffd61c0b2afe55377f53f3b994656
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e16e13a4280f02a538d636c49c72ba72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4db647606e6d9f5ff07e67b727197e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e16e13a4280f02a538d636c49c72ba72
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32e33dfe42ad4d5242a8d1f84376f4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5532d1565aef070205a5dc643dd8ac6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6b60777e0c8bcdb18942f9b17a90beb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4cb61e5695c833be589bea926e9f83a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6b60777e0c8bcdb18942f9b17a90beb
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d1cee447dbb0d0e4b09ca13c44c09584(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 92, 140], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68f9591a5a2bc74555be4a16f834f96e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1cee447dbb0d0e4b09ca13c44c09584
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 12, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b84f85ff54bf888a0c4938109a97381(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccc5ce46b41706c4c6346070fabcd98d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b84f85ff54bf888a0c4938109a97381
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_447a102c8c794c72fb78f0466a2d82a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c151052102c3491c412426e5ddc94da0
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_12c5ea40abd90cc2166ea1b8d5b05243(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 300, 300], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_403323764c243e9c62c7ff03616328b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12c5ea40abd90cc2166ea1b8d5b05243
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_403323764c243e9c62c7ff03616328b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12c5ea40abd90cc2166ea1b8d5b05243
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6b7e35d2b8bb6782fe289b5b02c47e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 150, 150], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f39ec8f9816992e6f14f8b0af8d834c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6b7e35d2b8bb6782fe289b5b02c47e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f39ec8f9816992e6f14f8b0af8d834c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6b7e35d2b8bb6782fe289b5b02c47e5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4647331d53cc2e49c4014833f25fb29f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 75, 75], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a332f34cc2492bb5f66e254f0db50dc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4647331d53cc2e49c4014833f25fb29f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a332f34cc2492bb5f66e254f0db50dc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4647331d53cc2e49c4014833f25fb29f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a332f34cc2492bb5f66e254f0db50dc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4647331d53cc2e49c4014833f25fb29f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e7dfac4e1cda8b3055193547d195bbd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 38, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55491ddee9388501b28a88c5a271921b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7dfac4e1cda8b3055193547d195bbd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55491ddee9388501b28a88c5a271921b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7dfac4e1cda8b3055193547d195bbd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55491ddee9388501b28a88c5a271921b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7dfac4e1cda8b3055193547d195bbd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_af45e276b5165a4191909adff748a0dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 19, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd1d6f3edd896ac379229a5080516822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af45e276b5165a4191909adff748a0dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd1d6f3edd896ac379229a5080516822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af45e276b5165a4191909adff748a0dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd1d6f3edd896ac379229a5080516822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af45e276b5165a4191909adff748a0dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_acce54c7e92045eeccdd881f752814d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 19, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dcb47d337d30e2f05190320e5c4b274b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acce54c7e92045eeccdd881f752814d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcb47d337d30e2f05190320e5c4b274b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acce54c7e92045eeccdd881f752814d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6aab6a088faed72c1cb89ce81943b7a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 19, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d0006dda0e1b8ea7344ebac9542f3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6aab6a088faed72c1cb89ce81943b7a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_93f1ab01159a660be5d930ba50f181ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 10, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_384f01f7eff2bb1ffab16c7ec87fc5b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93f1ab01159a660be5d930ba50f181ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_301bd72c7b989eafdfb16cdf1097d9ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 10, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_826afbdc3050f3c46dd85075db504649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_301bd72c7b989eafdfb16cdf1097d9ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48734480f0635ba92002e8d9e8100740(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d66767274aa331fee61ce0840b623496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48734480f0635ba92002e8d9e8100740
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1bb8cef3468f0633d430012a09afb80f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b6d4feb3731d5523bf8b90ad8711dd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bb8cef3468f0633d430012a09afb80f
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bad1fada7d2eaf4a97ac87f253de56c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2970b5f8d5b58ca4975ceeca0b33090(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bad1fada7d2eaf4a97ac87f253de56c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3bc7e0b180bdea91048c1410e0ac9ca1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1a86115baeb91ae5b1e865b432bc977(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bc7e0b180bdea91048c1410e0ac9ca1
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4491457c985e51b3d09e09b1b226c9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2438751c68184df01388eee1c02f89b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4491457c985e51b3d09e09b1b226c9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c10181805adc89cee462e9bcbf03a1ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_79370a1afac5d32b4709985e864f1493(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c10181805adc89cee462e9bcbf03a1ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_edee3bc938a0f406632facec1f0b28aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c60d550ea9c074d9c78b92d402a4020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edee3bc938a0f406632facec1f0b28aa
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4801efee2c19228a8cbaacdd30aede6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2beb8a3e2b626fb2282be1d3434e1694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4801efee2c19228a8cbaacdd30aede6
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e5aaadc58c1cea081618ce405ddd072(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_771d35a7cc4fc4661f257763e5d11318(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e5aaadc58c1cea081618ce405ddd072
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_081b961a95d1219d6ee158b0750ff8fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.495512008666992]], [[5.432677745819092]], [[4.943984508514404]], [[3.9269022941589355]], [[4.808197498321533]], [[5.317215442657471]], [[4.479256629943848]], [[5.208613395690918]], [[4.893153190612793]], [[4.050814151763916]], [[4.705148696899414]], [[5.471085071563721]], [[4.391146659851074]], [[4.680963516235352]], [[4.875914573669434]], [[4.115333557128906]], [[4.703266620635986]], [[4.208454132080078]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_0d6784961e85cb5e80efafbadcc41c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a9c36a1e5b6752923abfacd8ee2892d
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccd100e22eaee704c29d5781a9dcf51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dba17fc8a3d4e07e1f074d3039d89933
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5f56eea8191ecd3067ac0dd4c885fd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40d9bd71ee55816c8107b466c3cc9836(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f56eea8191ecd3067ac0dd4c885fd8
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa4e8da2c05303e6d58f9ce76c2d8f87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6610159873962402]], [[4.382030487060547]], [[4.38055944442749]], [[4.22531270980835]], [[4.511491298675537]], [[4.1695027351379395]], [[3.865060806274414]], [[4.202451229095459]], [[4.938055515289307]], [[3.9300243854522705]], [[4.440571308135986]], [[3.968632459640503]], [[4.592230319976807]], [[3.6639225482940674]], [[4.616931915283203]], [[4.238045692443848]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    
    class PrimitiveOp_73f8c05177f8585f2fe55b33c4837401(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_300d301b1c0b651c70c764a7d9c50a09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73f8c05177f8585f2fe55b33c4837401
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72e58a0472ca507a2bfc43b735b43535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.4651007652282715]], [[4.632149696350098]], [[4.672952651977539]], [[4.845834732055664]], [[4.880059242248535]], [[4.666262626647949]], [[4.770022392272949]], [[4.779326438903809]], [[5.001338958740234]], [[4.0377397537231445]], [[4.39913272857666]], [[4.714983940124512]], [[4.946009635925293]], [[4.713995456695557]], [[5.021440505981445]], [[4.741600513458252]], [[4.7320637702941895]], [[4.2691755294799805]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_7392ab5b100a784acea6cd2210d72c70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9a5be699233ae66d2644ef9ba39603
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.449842095375061]], [[1.7916018962860107]], [[1.7100330591201782]], [[1.0517908334732056]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    
    class PrimitiveOp_a0e9e37a51c3eb33913eb0014bb1a79e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 109, 109], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a48015a7835664bd303df014a4d2077(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0e9e37a51c3eb33913eb0014bb1a79e
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7fa76e59a6cbba253350a03f127f869f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 16, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b6425010aa9f097d2a29ea11efa75c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fa76e59a6cbba253350a03f127f869f
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5bd109bcb434334dc7ff51534c7afadd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 64, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0508e2ecf849d5f5f7c0a81d23734e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5bd109bcb434334dc7ff51534c7afadd
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0508e2ecf849d5f5f7c0a81d23734e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5bd109bcb434334dc7ff51534c7afadd
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b6425010aa9f097d2a29ea11efa75c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7fa76e59a6cbba253350a03f127f869f
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0508e2ecf849d5f5f7c0a81d23734e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5bd109bcb434334dc7ff51534c7afadd
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0508e2ecf849d5f5f7c0a81d23734e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5bd109bcb434334dc7ff51534c7afadd
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd98837b9f719e1e3c3122855ba1ea45(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 32, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c434161e284f9cecf0708f5d2947471d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd98837b9f719e1e3c3122855ba1ea45
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a33730f6723d487094cda135f7cf1cf9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 128, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_986ace31d8360cd7e1fe87784f3c0270(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33730f6723d487094cda135f7cf1cf9
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_986ace31d8360cd7e1fe87784f3c0270(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33730f6723d487094cda135f7cf1cf9
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86ebad239dc4a6e44e3e000810ca162e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 32, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_186602963e2e1c9a1d8a6fa931aa3b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86ebad239dc4a6e44e3e000810ca162e
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_758583345dfb4f0684f1b878bfb5a6ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 128, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbc32731f9284c012b16a508fb1fd25e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758583345dfb4f0684f1b878bfb5a6ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbc32731f9284c012b16a508fb1fd25e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758583345dfb4f0684f1b878bfb5a6ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_427e5127369ab012b14f1c7781266cff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 48, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_acc214cab29e41ed3789cd14b2fcff36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_427e5127369ab012b14f1c7781266cff
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d35c598a257340de1a2b02d0bb7dc5fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dba7b1ce4ed33155767ea68945b4c2ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d35c598a257340de1a2b02d0bb7dc5fc
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dba7b1ce4ed33155767ea68945b4c2ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d35c598a257340de1a2b02d0bb7dc5fc
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acc214cab29e41ed3789cd14b2fcff36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_427e5127369ab012b14f1c7781266cff
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dba7b1ce4ed33155767ea68945b4c2ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d35c598a257340de1a2b02d0bb7dc5fc
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dba7b1ce4ed33155767ea68945b4c2ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d35c598a257340de1a2b02d0bb7dc5fc
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a2813a21125b83f18f0f51cbae6a89e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 64, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71be02e1ff300344e30982fdf9759036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2813a21125b83f18f0f51cbae6a89e0
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba180821f985e533707c30dcdc1a0fa8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 256, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_187b607e5c5d006a20c8a0a02349b386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba180821f985e533707c30dcdc1a0fa8
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_187b607e5c5d006a20c8a0a02349b386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba180821f985e533707c30dcdc1a0fa8
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_216a3284c6060d76e38b5ccf392775eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 64, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18b3efa7c591bbe98319a563b6ecfd03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_216a3284c6060d76e38b5ccf392775eb
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3cff2c664ba09a12437cd7aa0dd41e61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 256, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e64bf821fcb8c4742d9f586faec8e09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cff2c664ba09a12437cd7aa0dd41e61
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e64bf821fcb8c4742d9f586faec8e09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cff2c664ba09a12437cd7aa0dd41e61
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3fd4e6fe6b99ffdec563ac31ea59681(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1000, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d5243eccbed57a50585c6d8e5299e86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3fd4e6fe6b99ffdec563ac31ea59681
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d6784961e85cb5e80efafbadcc41c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a9c36a1e5b6752923abfacd8ee2892d
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_444ffcc16bcc6ef8b6478fc2c52a8fa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f85328aac18b5531513e7e776d667093
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ae7023e6f0a6653555e30d4c5f8f8c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c8c775d3f5cc23d87d46f0aadda825e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ae7023e6f0a6653555e30d4c5f8f8c3
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c989907d8e2984a0b9245caaf93b352f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_904357ed9d3a776389ccb16f3ac6200d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c989907d8e2984a0b9245caaf93b352f
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d72b5010f956129291db9c43a6985fa7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b5d4db7e2f95b31cb82efe470b18125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d72b5010f956129291db9c43a6985fa7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4cb61e5695c833be589bea926e9f83a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6b60777e0c8bcdb18942f9b17a90beb
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d6784961e85cb5e80efafbadcc41c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a9c36a1e5b6752923abfacd8ee2892d
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d24da0d46548f30fe843556043ec2aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9d9bcc1ddd3729d465db05074aabcec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e726a5cd289b8248d7cdf0b5ec8b181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.5449934005737305]], [[5.682139873504639]], [[5.67764949798584]], [[5.4259419441223145]], [[6.047888278961182]], [[5.0137834548950195]], [[5.673959255218506]], [[5.87434720993042]], [[6.007184028625488]], [[5.486945629119873]], [[5.58497428894043]], [[5.264822006225586]], [[5.091436862945557]], [[6.080480098724365]], [[5.848564624786377]], [[5.892393112182617]], [[6.222475051879883]], [[5.960496425628662]], [[5.735707759857178]], [[6.294724464416504]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_58072d31b54cc2f8dd9d1fcd35c802da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 84, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0b4ccf941d86e3ec314f5b53197faf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58072d31b54cc2f8dd9d1fcd35c802da
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e113884da24056f03867a9ce2a2112a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebaedbd1ee5b508bed9ae0f1cfda3fbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.1557114124298096]], [[2.9917681217193604]], [[3.779399871826172]], [[3.1832642555236816]], [[3.430224657058716]], [[3.3582873344421387]], [[2.883920669555664]], [[3.1816792488098145]], [[3.256134510040283]], [[3.319467306137085]], [[3.4573159217834473]], [[3.527083396911621]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_25fb5994312c796582cddb484994f99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.775596618652344]], [[4.627426624298096]], [[5.101263999938965]], [[5.359864711761475]], [[5.436671257019043]], [[6.06939697265625]], [[5.455032825469971]], [[5.321661949157715]], [[5.425404071807861]], [[5.160300254821777]], [[5.353097438812256]], [[5.381292819976807]], [[4.7527241706848145]], [[5.453306198120117]], [[5.433315277099609]], [[5.309089660644531]], [[4.748722076416016]], [[5.337813377380371]], [[5.172105312347412]], [[5.770773410797119]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_f75059d5ad295cecff04e3ee18c0829b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85b01dc5b87514b414a7a8aee3b35d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.358640193939209]], [[2.7018392086029053]], [[2.984464168548584]], [[2.702528953552246]], [[3.595785140991211]], [[2.438364267349243]], [[2.806942939758301]], [[3.076977252960205]], [[2.4327540397644043]], [[3.409235954284668]], [[2.8855111598968506]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b5d4db7e2f95b31cb82efe470b18125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d72b5010f956129291db9c43a6985fa7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e3eb51b9d288030f339d849d6ce6c4a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 56, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d46850e640e844f247cc49e78e21ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e3eb51b9d288030f339d849d6ce6c4a
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_96dc8643dc29e249b7d4dda0732345c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 14, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb8dcb47bc4d8df50f290db25d494b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96dc8643dc29e249b7d4dda0732345c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.03779935836792]], [[3.275832176208496]], [[2.847233772277832]], [[3.191497802734375]], [[3.2837717533111572]], [[3.0297393798828125]], [[3.781230926513672]], [[3.339047908782959]], [[3.7403409481048584]], [[3.3798489570617676]], [[4.031096935272217]], [[3.373056173324585]], [[3.172370195388794]], [[3.249523162841797]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    
    class PrimitiveOp_21261e3dfb0b121627aedc0e019a2d32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cecaed3364a965c507b2e0c4f4305b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21261e3dfb0b121627aedc0e019a2d32
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6757ea9a9b63f49de33939ad09e6eb4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c20d892bd43816678cbde0e09c267bb1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97f1c1751a2f9665684ad29f8c9cde3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.5319061279296875]], [[5.019760608673096]], [[5.33010196685791]], [[5.282497882843018]], [[5.046922206878662]], [[5.342513561248779]], [[5.555017471313477]], [[5.292306900024414]], [[4.68528938293457]], [[4.734997272491455]], [[5.910179615020752]], [[4.748666286468506]], [[5.971632957458496]], [[5.610820770263672]], [[4.633826732635498]], [[5.679800510406494]], [[6.121128082275391]], [[5.734371662139893]], [[4.891031265258789]], [[5.260383129119873]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_cb1f74e12d32ec89b81a4d654d13ef6e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33a048f17d0c226945783f34b1d3b32a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb1f74e12d32ec89b81a4d654d13ef6e
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33a048f17d0c226945783f34b1d3b32a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb1f74e12d32ec89b81a4d654d13ef6e
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33a048f17d0c226945783f34b1d3b32a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb1f74e12d32ec89b81a4d654d13ef6e
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33a048f17d0c226945783f34b1d3b32a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb1f74e12d32ec89b81a4d654d13ef6e
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_56177490843e69977abb19362dd06d6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eda28d032a5f635d0ed7ce0999a6e322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34974.4765625]], [[39784.2109375]], [[28418.41796875]], [[34156.3203125]], [[35816.92578125]], [[27483.953125]]], [[[34581.046875]], [[39333.0859375]], [[28099.552734375]], [[33769.2890625]], [[35419.4296875]], [[27170.34765625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_f4d5db634cae911e6512c8abbb2b2b67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[39317.140625]], [[33706.33203125]], [[41887.5703125]], [[37642.73828125]], [[40813.0546875]], [[41225.6953125]]], [[[40262.890625]], [[34524.9296875]], [[42899.82421875]], [[38544.11328125]], [[41793.37109375]], [[42220.3203125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_45ff2a4d2762e278893e7afbb6c37bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[39998.03125]], [[36421.7578125]], [[39325.65625]], [[47435.87109375]], [[31697.357421875]], [[40192.0703125]]], [[[41765.7421875]], [[38032.9765625]], [[41064.15234375]], [[49526.53515625]], [[33105.390625]], [[41965.94140625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_1737a29299ab6864a7fc74e8e1ebc11d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41410.9140625]], [[47603.7734375]], [[45140.16796875]], [[48028.58203125]], [[42334.01953125]], [[38568.15234375]]], [[[43256.73828125]], [[49733.0703125]], [[47160.5]], [[50171.8359375]], [[44224.015625]], [[40284.5]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    
    class PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f35e43bfd2bd8941a87a4b8cd4f9cf9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebc6376d12c3bc818667a379e4bb1aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd258301bd7b69c64e2f338a971a38fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349130a2b3ae65d4d413b9ad920510cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_135c1747edb747b0c5429af97e7fe642(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf665ce1f31dd9f2ce867dd6cf337f89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135c1747edb747b0c5429af97e7fe642
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3130d36ead250381bb017004e21b24d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 6, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d24da0d46548f30fe843556043ec2aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9d9bcc1ddd3729d465db05074aabcec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89fa8ade2ecf3bcc0e0798912005d3d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.444602012634277]], [[7.261509418487549]], [[8.290815353393555]], [[6.83220100402832]], [[7.594128131866455]], [[7.976133346557617]], [[7.4478888511657715]], [[6.68143367767334]], [[8.074603080749512]], [[7.364259719848633]], [[7.592048645019531]], [[7.421213626861572]], [[7.4837775230407715]], [[7.403409004211426]], [[8.940410614013672]], [[7.858551025390625]], [[6.976071834564209]], [[7.378269195556641]], [[7.4309563636779785]], [[7.936702251434326]], [[7.602663993835449]], [[7.9230146408081055]], [[7.234231472015381]], [[7.327571868896484]], [[7.961794376373291]], [[8.160552978515625]], [[8.463578224182129]], [[8.798234939575195]], [[8.465989112854004]], [[7.31309700012207]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_94b36d4f50d73c2c017b9cdac5f44d9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.720489501953125]], [[7.950204849243164]], [[8.676801681518555]], [[8.149105072021484]], [[7.102902412414551]], [[7.874119758605957]], [[8.240650177001953]], [[7.847740173339844]], [[7.982648849487305]], [[7.491360664367676]], [[8.735313415527344]], [[7.510238170623779]], [[8.066055297851562]], [[8.007878303527832]], [[8.85459041595459]], [[8.496610641479492]], [[8.094125747680664]], [[7.891778945922852]], [[8.22463607788086]], [[7.96538782119751]], [[7.587237358093262]], [[8.23629379272461]], [[8.250158309936523]], [[8.19361400604248]], [[8.194456100463867]], [[9.199542045593262]], [[7.542173385620117]], [[8.364714622497559]], [[7.895327091217041]], [[8.275627136230469]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_389cec700bf67c34bcc65127afe799cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 44, 66], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9eb14cea34f3db6d98e852ef818593b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_389cec700bf67c34bcc65127afe799cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_107ffe929d492ec2d9f571c100c1aaad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.00590991973877]], [[7.415954113006592]], [[7.876989841461182]], [[6.942861080169678]], [[8.300236701965332]], [[7.491955757141113]], [[7.546222686767578]], [[7.608494758605957]], [[7.794553279876709]], [[7.298939228057861]], [[8.410886764526367]], [[7.41102409362793]], [[8.027511596679688]], [[7.821455478668213]], [[8.39918041229248]], [[8.598889350891113]], [[8.43173885345459]], [[8.188957214355469]], [[8.038849830627441]], [[7.986676216125488]], [[7.033593654632568]], [[7.4206156730651855]], [[8.053648948669434]], [[7.339644908905029]], [[7.243391513824463]], [[8.141550064086914]], [[7.59982967376709]], [[8.755999565124512]], [[8.071479797363281]], [[7.06417179107666]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    
    class PrimitiveOp_c40f995f18da3af53bb053bca14a33ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 50, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9aa401fac87af806eb5714cee12a63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40f995f18da3af53bb053bca14a33ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d6784961e85cb5e80efafbadcc41c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a9c36a1e5b6752923abfacd8ee2892d
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41c5af3f87508f81fda37264012b9f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.14487075805664]], [[8.147989273071289]], [[8.095406532287598]], [[7.036656856536865]], [[7.791210174560547]], [[7.92747688293457]], [[7.291300296783447]], [[7.665894031524658]], [[8.080425262451172]], [[7.467463970184326]], [[7.3757219314575195]], [[7.0921525955200195]], [[6.8974761962890625]], [[7.289829730987549]], [[7.877655982971191]], [[7.310974597930908]], [[7.013061046600342]], [[6.738163948059082]], [[7.917231559753418]], [[6.857556343078613]], [[8.298291206359863]], [[7.9575042724609375]], [[7.84797477722168]], [[8.23776912689209]], [[7.439065933227539]], [[8.512922286987305]], [[7.981821060180664]], [[7.359304428100586]], [[6.39490270614624]], [[7.269107818603516]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_4c0514f2d58a71a229cad155a5d4fbc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0138115882873535]], [[3.2304744720458984]], [[3.1001744270324707]], [[3.620997667312622]], [[3.425168514251709]], [[3.1345767974853516]], [[3.0422685146331787]], [[3.525670289993286]], [[3.386475086212158]], [[3.072099447250366]], [[3.4267210960388184]], [[2.8959836959838867]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_dc07fc2d3adc9e7002d68e5f721aa451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.5646629333496094]], [[3.4088218212127686]], [[4.021111011505127]], [[2.847135305404663]], [[3.5908870697021484]], [[2.934852361679077]], [[3.508193016052246]], [[3.6472582817077637]], [[3.661288261413574]], [[3.0511858463287354]], [[3.5024564266204834]], [[3.0943455696105957]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_a9cd47709a605d809dca59b7bd3429cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.729383945465088]], [[7.137293338775635]], [[6.083255290985107]], [[6.352455139160156]], [[7.294790267944336]], [[7.118271827697754]], [[7.140522003173828]], [[7.952057838439941]], [[7.3802666664123535]], [[7.0913896560668945]], [[6.739432334899902]], [[7.589100360870361]], [[7.090548515319824]], [[6.952294826507568]], [[7.234918594360352]], [[7.350093364715576]], [[6.242755889892578]], [[6.655361175537109]], [[5.618881702423096]], [[7.279906272888184]], [[6.899659633636475]], [[7.120492935180664]], [[7.67257022857666]], [[7.347502708435059]], [[6.531886100769043]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    
    class PrimitiveOp_30177369e9c7c730ab6716fc4e4d9091(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0aabe37b035579763669e2086e16169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30177369e9c7c730ab6716fc4e4d9091
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_edb28eeba3177ab4c46a165d33e1ccdc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 312], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bba69a7d904b8bf85c43754193ad22f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edb28eeba3177ab4c46a165d33e1ccdc
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_75f860974925537ff731a845d6597e4d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d41209d23dcf93ffed998a94e3519d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75f860974925537ff731a845d6597e4d
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9d0566dfe988a450cde1c220c62e4c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17ff03d3bffd90dc7f6b8e8286765306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9d0566dfe988a450cde1c220c62e4c0
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3882ef617fa2901f15036c6def9adbfa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 5, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de4e3a8c4dc9c6599fb5ae505977413c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3882ef617fa2901f15036c6def9adbfa
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4c629923e81c3422b6e1e0a4a83f9ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.62026834487915]], [[4.912600517272949]], [[4.501530170440674]], [[5.113858699798584]], [[5.229439735412598]], [[4.599358081817627]], [[5.316549777984619]], [[4.630554676055908]], [[4.967809200286865]], [[4.93235969543457]], [[5.05343770980835]], [[5.221227169036865]], [[4.371150493621826]], [[5.340354919433594]], [[5.351968288421631]], [[5.038957118988037]], [[4.813811779022217]], [[5.55605411529541]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_8c273a15fd86d42676f746edcaed6708(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 39], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a097b271db25b0fcfd1f03c4b62ef474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c273a15fd86d42676f746edcaed6708
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8f86f432645babc0a68a941e70343e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3406821489334106]], [[1.40310800075531]], [[1.3875422477722168]], [[1.3541191816329956]], [[1.5349763631820679]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_b0ab310ddba0dcb00ad2b581937fb05f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.891082525253296]], [[2.8003101348876953]], [[3.0362467765808105]], [[3.3644373416900635]], [[2.7299394607543945]], [[3.5558624267578125]], [[3.410123109817505]], [[3.1943938732147217]], [[3.5989999771118164]], [[2.9150657653808594]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_5e95413a8c45003faab305175bbe4179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.823153972625732]], [[5.129012584686279]], [[5.525485515594482]], [[4.958107948303223]], [[5.484551906585693]], [[4.751092910766602]], [[5.324677467346191]], [[6.043829917907715]], [[5.8095316886901855]], [[5.279562473297119]], [[5.300936222076416]], [[6.204381465911865]], [[5.513743877410889]], [[5.982616424560547]], [[5.167477607727051]], [[5.536433219909668]], [[4.966822147369385]], [[5.783491134643555]], [[4.380859851837158]], [[5.523407459259033]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9aa401fac87af806eb5714cee12a63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40f995f18da3af53bb053bca14a33ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30706f1944ebd6943d0de63e07ff03b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f7bf0bc2afb7d4921ef4cd868b8bbea
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6d84af4fe1e2daadbfda76d384aab2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3eab7a4665cb38068f08b87124ed65e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f605ef3c9d5aba83a2bfe91034f1e23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.116128921508789]], [[6.4880290031433105]], [[6.603219032287598]], [[5.883945465087891]], [[5.97711706161499]], [[6.399841785430908]], [[6.568505764007568]], [[7.100306987762451]], [[6.596995830535889]], [[6.940871238708496]], [[6.557913780212402]], [[6.544834613800049]], [[5.847147464752197]], [[7.082651138305664]], [[6.433468341827393]], [[6.7771406173706055]], [[6.6440534591674805]], [[6.463765621185303]], [[6.267218112945557]], [[6.189576625823975]], [[7.43379020690918]], [[6.055135250091553]], [[6.613389492034912]], [[6.876086711883545]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    
    class PrimitiveOp_756614e5418c87ec5de777efd1a9f45e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e145857af4d933b68fdd914cc0d67e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_756614e5418c87ec5de777efd1a9f45e
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b47a13e096b21ef1f75d7943d52d5bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0780367851257324]], [[2.8307974338531494]], [[2.5395758152008057]], [[2.5772764682769775]], [[3.0108275413513184]], [[3.0733754634857178]], [[2.7212250232696533]], [[3.425830125808716]], [[3.5829856395721436]], [[2.8356549739837646]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    
    class PrimitiveOp_f705a2fc1fd6aafecc99521c33692b06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0005866662478f70120d065aca6a6df6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f705a2fc1fd6aafecc99521c33692b06
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48caeea999c0c62afb099158fad48a2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 40, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8663667eed1ecf26789581aabf453026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48caeea999c0c62afb099158fad48a2f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8abf95a695fe6cf68198d7b278d0a23f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97ed7a90317f3f5f238086f72d9f7a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3030c24a0cd552b2d38a4789f62b2793(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6161da0747a85a6a19c4051ee66dd809(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3030c24a0cd552b2d38a4789f62b2793
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4dcac19684d9330c9911021a9f9bf333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.2628655433654785]], [[5.365011692047119]], [[5.038057804107666]], [[5.306423187255859]], [[5.208133697509766]], [[4.970277786254883]], [[5.335209369659424]], [[5.289772033691406]], [[4.597698211669922]], [[4.871645450592041]], [[4.971390724182129]], [[4.5013556480407715]], [[5.485369682312012]], [[4.997920036315918]], [[5.272904872894287]], [[4.238986968994141]], [[5.5695109367370605]], [[4.83064079284668]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    
    class PrimitiveOp_b1c2d4786102bcc3bb26974ba10e39c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9096c4084ff6b626f373732dfef4f1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c2d4786102bcc3bb26974ba10e39c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[8.870004653930664, 9.070971488952637, 8.6998291015625, 8.771418571472168, 8.77302074432373, 8.278258323669434, 9.320741653442383, 8.51421070098877, 8.791091918945312, 8.457852363586426, 9.51916790008545, 8.142207145690918, 8.19686508178711, 8.17872428894043, 8.666762351989746, 8.191591262817383, 9.497175216674805, 8.180706024169922, 8.71663761138916, 9.759553909301758, 8.672922134399414, 8.273152351379395, 8.350447654724121, 9.068215370178223, 8.876349449157715, 8.625724792480469, 9.934979438781738, 8.560579299926758, 8.065917015075684, 9.131192207336426]], dtype='float32').reshape([1, 30]),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_904357ed9d3a776389ccb16f3ac6200d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c989907d8e2984a0b9245caaf93b352f
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e6ac5c13a7f3ae9c33a3e47125bfdb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.780267715454102]], [[8.131998062133789]], [[8.939107894897461]], [[8.882832527160645]], [[9.453350067138672]], [[8.85619068145752]], [[8.147751808166504]], [[8.396537780761719]], [[8.775067329406738]], [[8.127144813537598]], [[8.213729858398438]], [[8.807175636291504]], [[8.307685852050781]], [[9.488317489624023]], [[7.320240497589111]], [[8.850549697875977]], [[8.130460739135742]], [[8.228346824645996]], [[8.956204414367676]], [[8.420777320861816]], [[8.549696922302246]], [[7.867123126983643]], [[8.182889938354492]], [[8.662212371826172]], [[7.8356781005859375]], [[8.064471244812012]], [[7.9256134033203125]], [[7.720449447631836]], [[9.509540557861328]], [[8.814489364624023]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_e20eb3ae18a2847462c43afd1d42f2f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.213860273361206]], [[0.7169217467308044]], [[1.3732637166976929]], [[0.9356551766395569]], [[0.9903314113616943]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_b8709c65f832a61bc0649e2be2eab4b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.469463348388672]], [[3.039804458618164]], [[3.2300972938537598]], [[2.7001450061798096]], [[2.6241466999053955]], [[2.543884515762329]], [[2.668975591659546]], [[1.845589518547058]], [[2.947535514831543]], [[2.841306686401367]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_dd00903340f451387a7d2d444ac273be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.91616153717041]], [[4.971972942352295]], [[4.818826675415039]], [[4.943068027496338]], [[4.595055103302002]], [[5.193109035491943]], [[4.6342620849609375]], [[4.798951625823975]], [[5.12747859954834]], [[4.536078929901123]], [[3.727442502975464]], [[4.675539016723633]], [[5.137940406799316]], [[5.504049777984619]], [[4.462343692779541]], [[4.524420738220215]], [[5.013905048370361]], [[5.048361778259277]], [[4.960768222808838]], [[5.547129154205322]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7369e4709ff42b6c699e99ffa687801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.84181809425354]], [[3.919459342956543]], [[3.943265914916992]], [[3.615178108215332]], [[4.343533515930176]], [[4.357351303100586]], [[3.5953638553619385]], [[4.600090503692627]], [[3.89821195602417]], [[4.208963871002197]], [[4.014341831207275]], [[4.141739368438721]], [[4.350880146026611]], [[4.032442569732666]], [[4.039931774139404]], [[4.096301555633545]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_32e33dfe42ad4d5242a8d1f84376f4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5532d1565aef070205a5dc643dd8ac6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e180c84ec8eeb755e66742effae38afe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 36, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_540bec8b07137f99fb4ab5f0b8772af0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e180c84ec8eeb755e66742effae38afe
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07500d844602bc86be17c230da8e1ee2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 92, 140], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c5148dc3829dd905d7a9f0ab7520abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07500d844602bc86be17c230da8e1ee2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 46, 70], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44eac71f8e3d5ab4f7d527b1e87388ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28b1a8524f5ab8c8d4e6ff817f5e3194
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4341738b673b240cd4dd23e47ecd4311(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 23, 35], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b4a94d2bb40ef768546ec7ac0eeaef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4341738b673b240cd4dd23e47ecd4311
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77228773e2152f86625455f024c275bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d2bccd3c8b5cfa43b668b9d8409c3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b8450f44c24820e1d3245139db7b649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3130d36ead250381bb017004e21b24d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0b4ccf941d86e3ec314f5b53197faf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58072d31b54cc2f8dd9d1fcd35c802da
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cecaed3364a965c507b2e0c4f4305b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21261e3dfb0b121627aedc0e019a2d32
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_692a0798462d7e6e0bd65996eac293f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96dc8643dc29e249b7d4dda0732345c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.8748505115509033]], [[3.848036527633667]], [[3.7055251598358154]], [[3.35248064994812]], [[2.893239974975586]], [[4.06261682510376]], [[3.7211146354675293]], [[4.051092624664307]], [[3.9451940059661865]], [[4.032584190368652]], [[4.147378921508789]], [[2.9157514572143555]], [[3.8111705780029297]], [[3.7591934204101562]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_6b56a31a03f1db9cfc54abc13f2e075d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.051861763000488]], [[5.754777431488037]], [[4.818517684936523]], [[4.752799034118652]], [[4.6054182052612305]], [[5.425618648529053]], [[5.631992340087891]], [[5.630918979644775]], [[5.270631313323975]], [[5.155634880065918]], [[5.055809020996094]], [[4.907738208770752]], [[5.484491348266602]], [[4.884391784667969]], [[5.892811298370361]], [[5.109838962554932]], [[5.606997489929199]], [[5.1103386878967285]], [[5.412006378173828]], [[4.9560112953186035]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_d65b954991d4711b72b956d68313e17b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 14, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_508ac3e2723848e9f4a156fe70ce09ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d65b954991d4711b72b956d68313e17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30706f1944ebd6943d0de63e07ff03b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f7bf0bc2afb7d4921ef4cd868b8bbea
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adb6cfc8b868ac915c6cca920f59dd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.228400707244873]], [[7.572210311889648]], [[7.720287799835205]], [[7.640965461730957]], [[6.785443305969238]], [[8.622623443603516]], [[8.482710838317871]], [[7.288590431213379]], [[7.2805681228637695]], [[8.149312973022461]], [[7.05112361907959]], [[8.065946578979492]], [[7.8982014656066895]], [[8.205387115478516]], [[8.06624698638916]], [[7.016504287719727]], [[8.207212448120117]], [[6.216163158416748]], [[7.654832363128662]], [[7.563879013061523]], [[7.733938217163086]], [[7.262387275695801]], [[7.852588176727295]], [[7.543629169464111]], [[8.282045364379883]], [[7.994214057922363]], [[7.4732866287231445]], [[8.085190773010254]], [[8.281085968017578]], [[7.778512477874756]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_540bec8b07137f99fb4ab5f0b8772af0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e180c84ec8eeb755e66742effae38afe
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae54dae87f9d8bf00c18369f30bed092(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 96, 109, 109], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89c1dcf69ccb9a96d182a8a4fb402379(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae54dae87f9d8bf00c18369f30bed092
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ee75e2094aebe3d2e0a60a126917ab0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c3806f7c0a0d08ddcdb0cefaa352478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ee75e2094aebe3d2e0a60a126917ab0
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_92603cc856aff48ae8a9eb417d1f7254(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 64, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff3b7407f96158ee063b984e6743eb01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92603cc856aff48ae8a9eb417d1f7254
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff3b7407f96158ee063b984e6743eb01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92603cc856aff48ae8a9eb417d1f7254
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c3806f7c0a0d08ddcdb0cefaa352478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ee75e2094aebe3d2e0a60a126917ab0
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff3b7407f96158ee063b984e6743eb01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92603cc856aff48ae8a9eb417d1f7254
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff3b7407f96158ee063b984e6743eb01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92603cc856aff48ae8a9eb417d1f7254
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7644c188e65f1c93bbf63ec4e84bc0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_532de04c1ad5f21f206d832a488ae3b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7644c188e65f1c93bbf63ec4e84bc0c
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d16c06161420f7a222fceb3b8765a86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 128, 54, 54], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d7094513ebc3002007055486cdf0aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d16c06161420f7a222fceb3b8765a86
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d7094513ebc3002007055486cdf0aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d16c06161420f7a222fceb3b8765a86
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d37b766036cff65dcd2545445ad1afa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7facf386957334d54dad7a4047f544cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d37b766036cff65dcd2545445ad1afa
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b557191fd16de01f06bb9cd5cad1c1ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 128, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f0aabdbf22dc7a46bbd206e30f09d08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b557191fd16de01f06bb9cd5cad1c1ba
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f0aabdbf22dc7a46bbd206e30f09d08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b557191fd16de01f06bb9cd5cad1c1ba
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d7fcbc94a616b3d32518ae3d34dcf43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 48, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b9b48d4a6d19eaf2227f19d4247605b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d7fcbc94a616b3d32518ae3d34dcf43
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_244fbd59c84456039e1f6f45ef53b517(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 192, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd31e171fb170d1b45d0dad1e55aec9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_244fbd59c84456039e1f6f45ef53b517
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd31e171fb170d1b45d0dad1e55aec9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_244fbd59c84456039e1f6f45ef53b517
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b9b48d4a6d19eaf2227f19d4247605b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d7fcbc94a616b3d32518ae3d34dcf43
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd31e171fb170d1b45d0dad1e55aec9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_244fbd59c84456039e1f6f45ef53b517
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd31e171fb170d1b45d0dad1e55aec9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_244fbd59c84456039e1f6f45ef53b517
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b643e2223dd09e3011492721d90e7bc2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 64, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3243a06b924b426e3176a2c384d85a07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b643e2223dd09e3011492721d90e7bc2
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4617a23c4f62f8d3861c1eb98ad81c9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 256, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25f6c3c55e7dbb6e39599c80d78c75a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4617a23c4f62f8d3861c1eb98ad81c9d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25f6c3c55e7dbb6e39599c80d78c75a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4617a23c4f62f8d3861c1eb98ad81c9d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed565f3308b826ce968092fb08ff0535(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 64, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34f1316873a8abdf8102d90bd00af3b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed565f3308b826ce968092fb08ff0535
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2f24298a9c547fbfcf1c8444e280685b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 256, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_702601694620fd38ee4075085dbfea34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f24298a9c547fbfcf1c8444e280685b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_702601694620fd38ee4075085dbfea34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f24298a9c547fbfcf1c8444e280685b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0dbc057cdcaa0961cb060270c5afb1b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1000, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b92cfe771132b8eed6f58fd9658c918a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dbc057cdcaa0961cb060270c5afb1b0
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9aa401fac87af806eb5714cee12a63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c40f995f18da3af53bb053bca14a33ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7e4ac9328a0b499d916ef29cac9bf1f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 10, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d8fe947cffb0ff52c48c45d77e94f02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e4ac9328a0b499d916ef29cac9bf1f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b5d4db7e2f95b31cb82efe470b18125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d72b5010f956129291db9c43a6985fa7
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_924fc4a74f497ceac9386dbe5aef8622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.9753828048706055]], [[6.003129005432129]], [[7.127122402191162]], [[5.840324401855469]], [[6.891082763671875]], [[6.126930236816406]], [[7.248678684234619]], [[6.783142566680908]], [[6.2587175369262695]], [[5.727454662322998]], [[7.110597133636475]], [[6.120436668395996]], [[7.036701202392578]], [[6.22518253326416]], [[7.063243389129639]], [[6.6329193115234375]], [[6.64615535736084]], [[5.935981750488281]], [[6.5288004875183105]], [[7.519911289215088]], [[6.803238391876221]], [[6.365416526794434]], [[5.566114902496338]], [[7.818721294403076]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f3e0bfa446b2cc12ac167d5b389e8fe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.455387115478516]], [[6.70116662979126]], [[6.931902885437012]], [[6.813783168792725]], [[7.750301361083984]], [[7.221365451812744]], [[6.216297626495361]], [[6.400246620178223]], [[7.3066630363464355]], [[7.40735387802124]], [[6.200910568237305]], [[7.3968400955200195]], [[6.916532039642334]], [[7.136009693145752]], [[7.048123359680176]], [[7.445611953735352]], [[6.951174259185791]], [[6.86937952041626]], [[7.565418243408203]], [[6.618014335632324]], [[7.1575422286987305]], [[6.011700630187988]], [[7.5044941902160645]], [[6.530730247497559]], [[6.473878383636475]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_b2e13933ece0d906c6caee8c00943c97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.961212396621704]], [[2.8444416522979736]], [[3.0302486419677734]], [[3.4965062141418457]], [[4.392430305480957]], [[3.4449360370635986]], [[4.060918807983398]], [[2.69504451751709]], [[3.8072237968444824]], [[3.5579631328582764]], [[3.595301628112793]], [[3.4865787029266357]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30706f1944ebd6943d0de63e07ff03b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f7bf0bc2afb7d4921ef4cd868b8bbea
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73948bc60f908208a8ef9af6c77c18c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1cc2480b2e46faa2271589fdc286850
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a89bb02d678a7727526fdbd760a8fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b735795aa7b39bbb1819f8ebb0214a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32e33dfe42ad4d5242a8d1f84376f4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5532d1565aef070205a5dc643dd8ac6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c517ec84bc044636d5b38599a54b9226(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 112, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b12e4d66d262ab6d73b344d4f34d580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c517ec84bc044636d5b38599a54b9226
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c473746199a9ec2d2b0f01daa41bccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a25c7b355f54cf118252117a9200253
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c023a308fc6e7227abb7b22f6d86f4cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1242487d21e96fb08cccfba8a9ca13e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b070068c43f0db419717aa7e1d1b001(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[706.5451049804688]], [[698.1089477539062]], [[733.5790405273438]], [[774.1317138671875]], [[702.3773803710938]], [[767.1760864257812]], [[719.3434448242188]], [[709.82080078125]], [[691.0403442382812]], [[681.920166015625]], [[771.4940795898438]], [[669.5833740234375]], [[718.261962890625]], [[709.01171875]], [[721.8114013671875]], [[735.890625]], [[706.6136474609375]], [[698.2988891601562]], [[718.6580200195312]], [[772.4765014648438]], [[658.5473022460938]], [[714.2171630859375]], [[651.621337890625]], [[754.21337890625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_93ecb85e77f68e4a5008b7660bb01aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[70.37177276611328]], [[75.7459487915039]], [[73.71358489990234]], [[74.7445068359375]], [[72.09110260009766]], [[75.07721710205078]], [[75.77609252929688]], [[74.45829010009766]], [[70.37089538574219]], [[78.90221405029297]], [[83.17375946044922]], [[70.34793090820312]], [[81.13215637207031]], [[73.42264556884766]], [[70.3205795288086]], [[73.20497131347656]], [[77.08792114257812]], [[72.10660552978516]], [[67.34578704833984]], [[75.58836364746094]], [[74.80601501464844]], [[75.42642211914062]], [[81.91889953613281]], [[68.68610382080078]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f680801bc8100f56ceb70ec350df7f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45.29734420776367]], [[51.2456169128418]], [[43.81867218017578]], [[44.45344924926758]], [[44.131744384765625]], [[45.768131256103516]], [[46.679420471191406]], [[42.54667663574219]], [[38.03548812866211]], [[47.30936813354492]], [[46.49201202392578]], [[46.001251220703125]], [[39.304664611816406]], [[44.54432678222656]], [[44.52653121948242]], [[44.796451568603516]], [[49.56315231323242]], [[45.860416412353516]], [[42.550559997558594]], [[45.2556266784668]], [[46.87994384765625]], [[43.06817626953125]], [[48.176002502441406]], [[40.50491714477539]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_df655b2891f13630337e9f783a1b5be9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[22.481063842773438]], [[22.24987030029297]], [[22.205080032348633]], [[23.617107391357422]], [[22.71398162841797]], [[22.377277374267578]], [[23.214012145996094]], [[21.080718994140625]], [[21.404701232910156]], [[20.12889289855957]], [[23.075468063354492]], [[21.405946731567383]], [[23.80116081237793]], [[23.009891510009766]], [[20.116384506225586]], [[22.51435661315918]], [[22.5460205078125]], [[23.645549774169922]], [[22.766191482543945]], [[21.108200073242188]], [[21.514690399169922]], [[21.464120864868164]], [[23.93003273010254]], [[21.577220916748047]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    
    class PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b09ac06e458511bbde8d92b529f99b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33464.0625]], [[34128.3203125]], [[34373.05859375]], [[28562.703125]], [[32737.2890625]], [[36944.26953125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_c71448f050af251fdaa85447b23905ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[40044.80078125]], [[37509.421875]], [[43190.484375]], [[37027.70703125]], [[37379.66015625]], [[33082.0859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_b2eac422ad470bb1df758e4f02b1020c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45157.8984375]], [[35525.29296875]], [[38019.31640625]], [[36812.3671875]], [[32309.341796875]], [[43327.96484375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_b06b0fd00244d09d9326c0a0bc008b0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[46890.375]], [[40730.390625]], [[41829.359375]], [[43960.64453125]], [[42763.0546875]], [[40935.3671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    
    class PrimitiveOp_0ddfceb83350abaa9e4aca779a2b1adf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 11, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_897756ef5c3cd4ce7756817ca1abc5d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ddfceb83350abaa9e4aca779a2b1adf
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0aabe37b035579763669e2086e16169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30177369e9c7c730ab6716fc4e4d9091
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a4cafcd0930250db7342f2ffb27444bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 88, 132], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72b439420cbeb7c347c0c92aa442e6cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4cafcd0930250db7342f2ffb27444bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b19400e1bc91f3dbaa39df99b9618d18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.723304271697998]], [[6.127195835113525]], [[5.915198802947998]], [[5.47707986831665]], [[5.763934135437012]], [[5.835768699645996]], [[6.205861568450928]], [[5.506523132324219]], [[5.576387405395508]], [[5.98674201965332]], [[5.432145595550537]], [[5.712860584259033]], [[5.408754825592041]], [[5.702744960784912]], [[6.343686103820801]], [[6.0185065269470215]], [[5.901707172393799]], [[5.767567157745361]], [[5.454860687255859]], [[5.4875054359436035]], [[6.7340288162231445]], [[6.339094161987305]], [[6.3797926902771]], [[5.385677337646484]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_74386d138e1af98b08abe37048d63ae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11be419fc9f85858794f580ec076980e
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_435a2fd25d79c7e55cd594aceba5119b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d74784250606790da6979fc49c6078d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_435a2fd25d79c7e55cd594aceba5119b
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b0c5c07932e7171fd740d39eb54e163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ec411f74f42d97cfc0fa40362d34ae5
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_35113bc0e82d84882f705d675eda32a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58b74cbc7dced4c7dfe2148be5f2f0c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_625806210b7bc905ec11eef79a288dca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.relu(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbd8fdfcf2b3e079f484cf86a6898d76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.626307010650635, 4.391082763671875, 4.806268215179443, 4.401785850524902, 4.369851112365723, 4.688978672027588, 4.786715984344482, 4.2709174156188965, 4.522144794464111, 4.13695764541626, 5.095004558563232, 4.644773006439209, 3.8188412189483643, 4.205009937286377, 4.518152236938477, 4.431371212005615, 4.864565849304199, 3.6953768730163574]], dtype='float32').reshape([1, 18]),
            ]


    class TestPrimitiveOp_2c3b9f807002946afda9f1e240ef24e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.27309513092041, 5.833820343017578, 5.503009796142578, 5.4817891120910645, 5.425642013549805, 6.097849369049072, 5.734645843505859, 5.922000885009766, 5.353610038757324, 5.361513137817383, 4.979347229003906, 6.077544212341309, 5.27100133895874, 5.504151821136475, 5.766942977905273, 5.257235527038574, 5.357554912567139, 5.529504299163818, 5.592182159423828, 5.393463611602783, 5.493752479553223, 6.035822868347168, 5.610565185546875]], dtype='float32').reshape([1, 23]),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_104e8c57667c6fc35554d9575066b7f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_003f240ea7e94805c2fce3b7c0fa5069(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37bfe53edb6b828f64fbdcbd8df3542a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcde380ff20a90c9e3be7f4593d06c53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcde380ff20a90c9e3be7f4593d06c53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5165bd6ad685f7b4fc27daf6091e4c2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.365582466125488]], [[7.731196403503418]], [[7.964057445526123]], [[6.985565662384033]], [[7.5866594314575195]], [[8.946654319763184]], [[7.98096227645874]], [[8.842916488647461]], [[7.250166893005371]], [[7.330620765686035]], [[7.9806976318359375]], [[8.2101411819458]], [[8.502208709716797]], [[7.358593940734863]], [[7.772006988525391]], [[7.207477569580078]], [[7.093987464904785]], [[8.185877799987793]], [[8.008213996887207]], [[7.6435675621032715]], [[7.370347499847412]], [[7.112212181091309]], [[7.433649063110352]], [[7.9769439697265625]], [[7.491349697113037]], [[8.110529899597168]], [[7.971155166625977]], [[8.05787467956543]], [[7.60438346862793]], [[7.9871745109558105]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_985f2e0fdb496dd46b3239326dfa0c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3fc43d75ead90331cf83b1aee760133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71e4f06d322dedab1ffba5298dc074d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c0a1048eaec5087aacb2fbe7ebf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f6965e568da34de3a7d4200547ba1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35bb423d86765d3b641231885233abdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.194438934326172]], [[8.588522911071777]], [[7.182609558105469]], [[7.648037433624268]], [[6.894355773925781]], [[6.827334880828857]], [[7.831111431121826]], [[7.7364959716796875]], [[8.895010948181152]], [[7.1844940185546875]], [[7.992562294006348]], [[7.785795211791992]], [[8.145564079284668]], [[8.470157623291016]], [[8.22587776184082]], [[7.272141933441162]], [[8.16726303100586]], [[8.357715606689453]], [[7.874889850616455]], [[7.3341898918151855]], [[7.798108100891113]], [[7.923760414123535]], [[8.171854972839355]], [[8.929399490356445]], [[6.554434299468994]], [[7.777000427246094]], [[8.502005577087402]], [[8.291655540466309]], [[8.365804672241211]], [[8.238239288330078]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_e100b13dd610a7f8edd1473fbce2a8a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6d9e2bb14ec82e1e692c9eae4e6d61d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5952028036117554]], [[1.1798752546310425]], [[1.684241533279419]], [[1.4642938375473022]], [[1.7384600639343262]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_c1299e35b6b43949184bf6aa88194464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.7241156101226807]], [[2.818718194961548]], [[3.44044828414917]], [[3.0282323360443115]], [[2.9832065105438232]], [[2.824711799621582]], [[3.593562126159668]], [[3.125070095062256]], [[2.591668128967285]], [[3.1199605464935303]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c9bd222b2ba332c9ed6daeeeda9733f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.417937755584717]], [[5.54685640335083]], [[6.385406494140625]], [[6.424187660217285]], [[6.714924335479736]], [[5.881267547607422]], [[5.836642742156982]], [[5.545405864715576]], [[6.1585845947265625]], [[6.060319900512695]], [[5.8800458908081055]], [[6.286401748657227]], [[6.463059902191162]], [[5.401740074157715]], [[5.785524368286133]], [[5.819664478302002]], [[6.0124735832214355]], [[6.33400297164917]], [[6.509230613708496]], [[6.19317626953125]], [[6.521552085876465]], [[6.344600677490234]], [[5.961433410644531]], [[6.324902534484863]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f6733f91e8667f6a6712fdd104c32efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c276eaef0adf8ba4a3e95707dc766236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_543d0cac3eeee8f84d85ff595bfc4ecc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b65510ee5ac1dd9e8bab45a636c40694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.443090915679932]], [[4.749655723571777]], [[3.8140125274658203]], [[4.799515247344971]], [[4.431580066680908]], [[4.186607837677002]], [[4.143815994262695]], [[4.519565582275391]], [[4.756207466125488]], [[4.297165393829346]], [[3.9445648193359375]], [[4.05826473236084]], [[4.993553638458252]], [[4.120225429534912]], [[4.165502071380615]], [[4.1748857498168945]], [[3.941878080368042]], [[4.543561935424805]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdc5b17bca1568a0252d7f7ff58fce49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.8558502197265625]], [[6.249524116516113]], [[5.586520195007324]], [[5.772902965545654]], [[6.590846538543701]], [[6.154374122619629]], [[6.037872314453125]], [[5.574150085449219]], [[5.730854034423828]], [[5.262627124786377]], [[5.772827625274658]], [[6.190662384033203]], [[5.621458053588867]], [[5.770803451538086]], [[5.573040962219238]], [[5.884099960327148]], [[5.568533897399902]], [[5.506991863250732]], [[6.336510181427002]], [[5.539333820343018]], [[5.555763244628906]], [[6.508022308349609]], [[5.37600564956665]], [[5.509634971618652]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_661b800fa7b4dcf6c02d320a8849130c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25c9f759d7999cc894a7d48ee28abefa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7779f907acb193ec69e619ced81bf8ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9399208426475525]], [[0.7881795167922974]], [[0.7001737952232361]], [[1.489819884300232]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_661b800fa7b4dcf6c02d320a8849130c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_523af1a479ad0a91b976e22bec7c7b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.274381637573242]], [[3.215075969696045]], [[2.314345121383667]], [[2.7561116218566895]], [[3.2190792560577393]], [[3.261092185974121]], [[2.7221601009368896]], [[2.923419237136841]], [[3.007906436920166]], [[3.288865566253662]], [[3.1404919624328613]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf36707205434e3038086206503edf4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.629995346069336]], [[7.557254314422607]], [[8.062764167785645]], [[8.629494667053223]], [[7.266910552978516]], [[7.732415199279785]], [[7.328917980194092]], [[7.593316555023193]], [[7.870702743530273]], [[7.665899753570557]], [[8.408510208129883]], [[7.446771144866943]], [[8.450179100036621]], [[8.21670913696289]], [[8.057311058044434]], [[8.340054512023926]], [[8.768411636352539]], [[8.051828384399414]], [[8.499567985534668]], [[8.646636009216309]], [[7.833431720733643]], [[7.5960564613342285]], [[8.436894416809082]], [[8.195596694946289]], [[8.104879379272461]], [[7.728816032409668]], [[8.08309268951416]], [[7.9412055015563965]], [[8.761932373046875]], [[8.15397834777832]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_838de16734609059ffb82d9f8301d528(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5508d2666b97b065aeede02c2fd7e05d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6034d9d430fc776c9480411c412411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.589501857757568]], [[3.9595940113067627]], [[4.664587020874023]], [[4.551024913787842]], [[4.109001636505127]], [[4.198220252990723]], [[4.432218551635742]], [[4.4817118644714355]], [[3.9227426052093506]], [[4.625372409820557]], [[3.8187661170959473]], [[4.21016788482666]], [[4.568661689758301]], [[4.094541072845459]], [[4.191646575927734]], [[4.210667133331299]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_89c1c6c5c3dca25036ce1dbe529cf47b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c796b51188fb5012eaacdb7b369c8aef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf5962a26d036cc87954470ce0d0fa99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d7d4fc3d810d1c0437a9448bc88ef2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdae9f44b8771e2ed3a9eb7d501c2fcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ad8e88dfbe306404163a497ebe4e71a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0b2512498824b7591b8dc743a0d3ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.331165313720703]], [[6.932826519012451]], [[6.909614086151123]], [[7.851978302001953]], [[7.19744348526001]], [[7.731747627258301]], [[6.7408857345581055]], [[7.960355281829834]], [[6.684189796447754]], [[7.306125164031982]], [[6.787485122680664]], [[6.810930252075195]], [[6.970196723937988]], [[7.652226448059082]], [[7.5926666259765625]], [[7.016533374786377]], [[6.701245307922363]], [[7.672969818115234]], [[7.558865547180176]], [[7.018603324890137]], [[7.665589332580566]], [[7.1204938888549805]], [[6.956162452697754]], [[7.111342906951904]], [[7.7293500900268555]], [[7.124806880950928]], [[7.486032962799072]], [[7.5760722160339355]], [[7.356403350830078]], [[7.904211521148682]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_cd59a66ddaffc92e4cd9b0eaddbd4565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7d5c9ba710769629ee23b3989586a25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c059c7eb7810696fcb041e1907917cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.2826313972473145]], [[6.334243297576904]], [[6.943402290344238]], [[6.860344886779785]], [[7.050297737121582]], [[7.861888408660889]], [[6.671663284301758]], [[6.760298728942871]], [[7.774744510650635]], [[7.529026985168457]], [[8.103219032287598]], [[6.956748008728027]], [[7.525053024291992]], [[6.947916030883789]], [[6.68371057510376]], [[7.102306365966797]], [[6.944980144500732]], [[6.7735161781311035]], [[7.2324910163879395]], [[6.96896505355835]], [[7.399407386779785]], [[6.78544807434082]], [[8.230155944824219]], [[6.772273540496826]], [[7.684123516082764]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44e5711acd041502fd6f4f3b3931bdbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_186934ba0baaaeee0bb224e1b85ba71a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3ad91578a7755c199a0e34b2185ce1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3ad91578a7755c199a0e34b2185ce1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6adad680251e062ce24d7f6a5d6ab51a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de1bde39c2e857057eb7aed401e3922a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7ac052cfffc1b519fec23ed042e24b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.324487686157227]], [[5.197154998779297]], [[5.39374303817749]], [[4.502297401428223]], [[5.0500264167785645]], [[5.456243515014648]], [[4.709370136260986]], [[5.073709487915039]], [[5.223665714263916]], [[5.386110782623291]], [[4.320528507232666]], [[5.036959171295166]], [[5.214683532714844]], [[5.159992694854736]], [[5.648549556732178]], [[5.104668140411377]], [[5.089097499847412]], [[4.991738319396973]], [[4.9874982833862305]], [[4.739211559295654]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd59a66ddaffc92e4cd9b0eaddbd4565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be2a7fca2329e4c894bbef4c730b9c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.766676902770996]], [[4.8825483322143555]], [[4.5546064376831055]], [[4.451656818389893]], [[4.954763889312744]], [[5.02230167388916]], [[4.508750915527344]], [[4.965527534484863]], [[4.956589698791504]], [[4.86350679397583]], [[4.453602313995361]], [[4.658884048461914]], [[4.835541725158691]], [[4.794821739196777]], [[4.3191351890563965]], [[4.349730968475342]], [[4.615575790405273]], [[5.2794270515441895]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_156d539948890880771b994333c17e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58b74cbc7dced4c7dfe2148be5f2f0c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e10445e86c3156af5a4b5d53308cd59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de2cee03963f7cfce5ea7311bb223ec0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e587f2ac0a6f2c58ce37a77349cfd3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a91b04d4848202f0bbc78f49bb816e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a91b04d4848202f0bbc78f49bb816e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e587f2ac0a6f2c58ce37a77349cfd3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a91b04d4848202f0bbc78f49bb816e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a91b04d4848202f0bbc78f49bb816e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2c147e98e4f20c7ebb7137a256c813f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1640ab93b16e268776af192313234d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1640ab93b16e268776af192313234d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcb5af3c13227a89a0e0fc510ffaab57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fae7a75d3d774b96bbdb51940c4bc1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fae7a75d3d774b96bbdb51940c4bc1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65ccb4041569c054b8c009a49185804c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b084037cab58455ad926e8144f1f476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b084037cab58455ad926e8144f1f476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65ccb4041569c054b8c009a49185804c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b084037cab58455ad926e8144f1f476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b084037cab58455ad926e8144f1f476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e30b3dd07ef8d30394dd156c03045cee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca4a65f196ef84327b87bda54e00ce79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca4a65f196ef84327b87bda54e00ce79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d554e1e79a80e7c169342caf4e821507(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb0a00c13a17eb08d6ccb387fd354e40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb0a00c13a17eb08d6ccb387fd354e40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a18ed7f2180e47a577303a3b8d712635(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de1bde39c2e857057eb7aed401e3922a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abbd4009a45c7901f92eea56eff30efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.547247409820557]], [[5.241771221160889]], [[4.631717205047607]], [[5.429015636444092]], [[4.983003616333008]], [[4.7412495613098145]], [[4.678460597991943]], [[5.05424165725708]], [[4.121486186981201]], [[4.833622932434082]], [[4.7639336585998535]], [[4.579180717468262]], [[4.328996181488037]], [[4.869755744934082]], [[5.027015686035156]], [[4.672824859619141]], [[4.301534652709961]], [[4.660597324371338]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_985f2e0fdb496dd46b3239326dfa0c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bdf29ce9d0e46c3ba35aaad4069b733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.95380163192749]], [[5.899272918701172]], [[6.4619035720825195]], [[6.993582725524902]], [[6.350670337677002]], [[6.611702919006348]], [[6.624584197998047]], [[6.251131534576416]], [[5.933366775512695]], [[5.4472479820251465]], [[6.299964904785156]], [[6.762087345123291]], [[6.120079040527344]], [[6.5732316970825195]], [[5.486151218414307]], [[6.62627649307251]], [[6.923728942871094]], [[6.560196399688721]], [[6.101138591766357]], [[6.05964994430542]], [[7.280458450317383]], [[6.875477313995361]], [[5.495212078094482]], [[6.6487884521484375]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_030909805c34712c06a25bccb75b4358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30fdd5de4cbc8687690a1f839cdef30a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.828707695007324]], [[4.706035137176514]], [[4.936190605163574]], [[4.7714033126831055]], [[5.0137248039245605]], [[4.696411609649658]], [[4.654387950897217]], [[4.914733409881592]], [[4.297507286071777]], [[4.808157444000244]], [[4.754739284515381]], [[4.652151107788086]], [[4.769132614135742]], [[5.319541931152344]], [[5.285849094390869]], [[4.965315818786621]], [[5.107652187347412]], [[5.249327659606934]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_37e94b8ec1e8c53e97e2c2e8dc032568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c81b203dfdb9c52f6c65094bebf57e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_797ad55ad1d9f4ae1fe52087ed71a946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.074566841125488]], [[5.675384044647217]], [[4.589181423187256]], [[5.357204914093018]], [[4.686102867126465]], [[5.177704811096191]], [[4.444948673248291]], [[4.822673320770264]], [[4.742612838745117]], [[4.711595058441162]], [[4.70556640625]], [[4.762353420257568]], [[5.579042434692383]], [[4.624544143676758]], [[5.49599027633667]], [[4.738476753234863]], [[5.141264915466309]], [[4.434568405151367]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_37e94b8ec1e8c53e97e2c2e8dc032568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2d81ae6c134b9eeacb526b5a3c9da89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66d9f05053a6e90caf23a10546a99312(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64113de0db286f804ef2fbc5654d0296(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e353d4a0c255879f1031511bf5f913a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ed4633ad94b495c3989739e6c41d3dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ed4633ad94b495c3989739e6c41d3dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e353d4a0c255879f1031511bf5f913a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ed4633ad94b495c3989739e6c41d3dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ed4633ad94b495c3989739e6c41d3dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_358b591da7879b5bf4a215cc6f336a32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a698be547cd2aa36699ec86ce995e340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a698be547cd2aa36699ec86ce995e340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b293fff2e52e2d04fce3077c3d98c2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e55356682aa13428ad98107849609537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e55356682aa13428ad98107849609537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf97402eaa9d4d2a5bdd911d631cee52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586951b757abce9597bc747d76c8802b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586951b757abce9597bc747d76c8802b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf97402eaa9d4d2a5bdd911d631cee52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586951b757abce9597bc747d76c8802b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586951b757abce9597bc747d76c8802b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_709a5537aaf40781896ee04ebcbbab28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6ff073bf2fb269038ebe8ca1e2e59f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6ff073bf2fb269038ebe8ca1e2e59f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ba9dde0a839cad2a1132d0c91f96de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1152bb8bb84f0ec4d98713ece8da726(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1152bb8bb84f0ec4d98713ece8da726(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0373fc6f151c5fef290d9428bd287e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_661a8a778f54c32af0f7e2bb85bd27b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ad8e88dfbe306404163a497ebe4e71a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a4094eb5ae5fe002069743742c7aad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a57c8e5b268045f532daa4e6296aea3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64b9d7ac759f1f975220d52e3f4a189d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6adad680251e062ce24d7f6a5d6ab51a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65b166f23a09e6075f1f2385987574b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65b166f23a09e6075f1f2385987574b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d4e3cb37bb739a26de9bdf98b8bad74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d4e3cb37bb739a26de9bdf98b8bad74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_237152e67b68847fa3d5e7b2dd149375(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_237152e67b68847fa3d5e7b2dd149375(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_237152e67b68847fa3d5e7b2dd149375(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bef66c725ee1953084d4fc2ad1fd080e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bef66c725ee1953084d4fc2ad1fd080e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bef66c725ee1953084d4fc2ad1fd080e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b90ae36654e57948399b0c07934ab149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b90ae36654e57948399b0c07934ab149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b90ae36654e57948399b0c07934ab149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d80507a797979458812cb845dc0b4730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d80507a797979458812cb845dc0b4730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa728fdca659fa2ffbb8adb04e9560dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f1658677cf3df87746a3c96c774b7fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5abaaf69e7c2448c4b96ecf10b1d97f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dc5d59f7ffda0f1426334e0e3e05a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2b41c8f21b6a59cef9017e0d6603aac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7228c474af85f369bf64a9cc227d1ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_edb2282ff1462890b06235d7e8fb9e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb7f078aed68fec7f48882c6918dc4c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_354925d2a234e2292463cc352eb92a03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_266f109d0c38a38845cdb4e876a76a66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8bb3cd12dd5f58d3defc71ac6d4732e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51f779684e12b68c17717aecebf2c0d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc6bd2aa535697d3b7d4e6b25439cd86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.495512008666992]], [[5.432677745819092]], [[4.943984508514404]], [[3.9269022941589355]], [[4.808197498321533]], [[5.317215442657471]], [[4.479256629943848]], [[5.208613395690918]], [[4.893153190612793]], [[4.050814151763916]], [[4.705148696899414]], [[5.471085071563721]], [[4.391146659851074]], [[4.680963516235352]], [[4.875914573669434]], [[4.115333557128906]], [[4.703266620635986]], [[4.208454132080078]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_de1bde39c2e857057eb7aed401e3922a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c435bfb3c1d972ebddd7c4bb2762cf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17f8cf3bdc8a573027549ccfc32058e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6af7c07a2db8d7ec7f6a7a887cae9b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6610159873962402]], [[4.382030487060547]], [[4.38055944442749]], [[4.22531270980835]], [[4.511491298675537]], [[4.1695027351379395]], [[3.865060806274414]], [[4.202451229095459]], [[4.938055515289307]], [[3.9300243854522705]], [[4.440571308135986]], [[3.968632459640503]], [[4.592230319976807]], [[3.6639225482940674]], [[4.616931915283203]], [[4.238045692443848]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_60bd3b12f7da8fa737596878586c26ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04a49e088ef1a1d2ed40baf0825ce3d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.4651007652282715]], [[4.632149696350098]], [[4.672952651977539]], [[4.845834732055664]], [[4.880059242248535]], [[4.666262626647949]], [[4.770022392272949]], [[4.779326438903809]], [[5.001338958740234]], [[4.0377397537231445]], [[4.39913272857666]], [[4.714983940124512]], [[4.946009635925293]], [[4.713995456695557]], [[5.021440505981445]], [[4.741600513458252]], [[4.7320637702941895]], [[4.2691755294799805]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_9d17c7afffa5a70258b3a6fdd0ac2c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.449842095375061]], [[1.7916018962860107]], [[1.7100330591201782]], [[1.0517908334732056]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_8b18c3ac2b492cd3a1184f13125c5734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9407771f661af4757d8e58bd7496bd7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b80ac8fa50e2f39cb39dd4565e929c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b80ac8fa50e2f39cb39dd4565e929c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9407771f661af4757d8e58bd7496bd7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b80ac8fa50e2f39cb39dd4565e929c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b80ac8fa50e2f39cb39dd4565e929c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_709f14e699ad8603e7f718a4fb12b5d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5481a28c314aaba99f68cb697330dc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5481a28c314aaba99f68cb697330dc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2dfd3d1e6d3ffc6ff143097bf62763e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57cb86f5cea72d6644a1cd1c4754a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d57cb86f5cea72d6644a1cd1c4754a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb396fdc4398e2ef93de7907e5f31ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb3d40812ec133e00c7b314195643bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb3d40812ec133e00c7b314195643bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb396fdc4398e2ef93de7907e5f31ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb3d40812ec133e00c7b314195643bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb3d40812ec133e00c7b314195643bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa8fe485d70f20b781be7ec02f1d6f55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7933f834b516f7f1b839d77d3eb09044(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7933f834b516f7f1b839d77d3eb09044(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b851c173127d3ac5e46dbd8632a17914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ea2ab0bd213815e7e541344b392e4c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ea2ab0bd213815e7e541344b392e4c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a816d366fe3c0f68b688ac805e9dacfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de1bde39c2e857057eb7aed401e3922a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_186934ba0baaaeee0bb224e1b85ba71a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4a80cab1a660c95aa832cc45f64f4e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973e03ebee097ba10130598c59d9fa41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04845b1dab159278b9ff8ce55f81ddfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a4094eb5ae5fe002069743742c7aad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de1bde39c2e857057eb7aed401e3922a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37e94b8ec1e8c53e97e2c2e8dc032568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a79d638b38ccc7e6059e09eb4f2e0247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.5449934005737305]], [[5.682139873504639]], [[5.67764949798584]], [[5.4259419441223145]], [[6.047888278961182]], [[5.0137834548950195]], [[5.673959255218506]], [[5.87434720993042]], [[6.007184028625488]], [[5.486945629119873]], [[5.58497428894043]], [[5.264822006225586]], [[5.091436862945557]], [[6.080480098724365]], [[5.848564624786377]], [[5.892393112182617]], [[6.222475051879883]], [[5.960496425628662]], [[5.735707759857178]], [[6.294724464416504]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_e341740e0f212bca7cbc0355e70c8765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ee44a0e7dfe2a507d6f65c4c3ab80d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.1557114124298096]], [[2.9917681217193604]], [[3.779399871826172]], [[3.1832642555236816]], [[3.430224657058716]], [[3.3582873344421387]], [[2.883920669555664]], [[3.1816792488098145]], [[3.256134510040283]], [[3.319467306137085]], [[3.4573159217834473]], [[3.527083396911621]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_5ce3189c0db4da12cc82726c79ba7ed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.775596618652344]], [[4.627426624298096]], [[5.101263999938965]], [[5.359864711761475]], [[5.436671257019043]], [[6.06939697265625]], [[5.455032825469971]], [[5.321661949157715]], [[5.425404071807861]], [[5.160300254821777]], [[5.353097438812256]], [[5.381292819976807]], [[4.7527241706848145]], [[5.453306198120117]], [[5.433315277099609]], [[5.309089660644531]], [[4.748722076416016]], [[5.337813377380371]], [[5.172105312347412]], [[5.770773410797119]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_3bf120f0b4a375f68fe6692baa778d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.358640193939209]], [[2.7018392086029053]], [[2.984464168548584]], [[2.702528953552246]], [[3.595785140991211]], [[2.438364267349243]], [[2.806942939758301]], [[3.076977252960205]], [[2.4327540397644043]], [[3.409235954284668]], [[2.8855111598968506]]]], dtype='float32').reshape([1, 11, 1, 1]),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04845b1dab159278b9ff8ce55f81ddfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e254a245e1baba7daacd6423dc265a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1222146334fa93b99f2aff980903914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.03779935836792]], [[3.275832176208496]], [[2.847233772277832]], [[3.191497802734375]], [[3.2837717533111572]], [[3.0297393798828125]], [[3.781230926513672]], [[3.339047908782959]], [[3.7403409481048584]], [[3.3798489570617676]], [[4.031096935272217]], [[3.373056173324585]], [[3.172370195388794]], [[3.249523162841797]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_7d3adbfd2eadaa6a75d16f583fd2691b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c276eaef0adf8ba4a3e95707dc766236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7066ff21a596a8b183cb00f8affd20d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.5319061279296875]], [[5.019760608673096]], [[5.33010196685791]], [[5.282497882843018]], [[5.046922206878662]], [[5.342513561248779]], [[5.555017471313477]], [[5.292306900024414]], [[4.68528938293457]], [[4.734997272491455]], [[5.910179615020752]], [[4.748666286468506]], [[5.971632957458496]], [[5.610820770263672]], [[4.633826732635498]], [[5.679800510406494]], [[6.121128082275391]], [[5.734371662139893]], [[4.891031265258789]], [[5.260383129119873]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_e7d303c3e5aafa0783314c2417fa39d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7d303c3e5aafa0783314c2417fa39d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7d303c3e5aafa0783314c2417fa39d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7d303c3e5aafa0783314c2417fa39d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b04df9ce826cef31922077176cba5b08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34974.4765625]], [[39784.2109375]], [[28418.41796875]], [[34156.3203125]], [[35816.92578125]], [[27483.953125]]], [[[34581.046875]], [[39333.0859375]], [[28099.552734375]], [[33769.2890625]], [[35419.4296875]], [[27170.34765625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_1f6f746e454c906cc7c4363c176484f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[39317.140625]], [[33706.33203125]], [[41887.5703125]], [[37642.73828125]], [[40813.0546875]], [[41225.6953125]]], [[[40262.890625]], [[34524.9296875]], [[42899.82421875]], [[38544.11328125]], [[41793.37109375]], [[42220.3203125]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_416dbde07770d0cd0d565bd8cba3d02f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[39998.03125]], [[36421.7578125]], [[39325.65625]], [[47435.87109375]], [[31697.357421875]], [[40192.0703125]]], [[[41765.7421875]], [[38032.9765625]], [[41064.15234375]], [[49526.53515625]], [[33105.390625]], [[41965.94140625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_95f3963684d76d2ba0a511322acbe916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41410.9140625]], [[47603.7734375]], [[45140.16796875]], [[48028.58203125]], [[42334.01953125]], [[38568.15234375]]], [[[43256.73828125]], [[49733.0703125]], [[47160.5]], [[50171.8359375]], [[44224.015625]], [[40284.5]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb32cf77941270b35e28f80f5a6ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c515607736d0dc8ad8b073d01d0c656b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2cce3cf597b783b0da988d647b47aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37e94b8ec1e8c53e97e2c2e8dc032568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_433415407eb5d67a9122a4c9861ad987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.444602012634277]], [[7.261509418487549]], [[8.290815353393555]], [[6.83220100402832]], [[7.594128131866455]], [[7.976133346557617]], [[7.4478888511657715]], [[6.68143367767334]], [[8.074603080749512]], [[7.364259719848633]], [[7.592048645019531]], [[7.421213626861572]], [[7.4837775230407715]], [[7.403409004211426]], [[8.940410614013672]], [[7.858551025390625]], [[6.976071834564209]], [[7.378269195556641]], [[7.4309563636779785]], [[7.936702251434326]], [[7.602663993835449]], [[7.9230146408081055]], [[7.234231472015381]], [[7.327571868896484]], [[7.961794376373291]], [[8.160552978515625]], [[8.463578224182129]], [[8.798234939575195]], [[8.465989112854004]], [[7.31309700012207]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_caa725833b2299d13bf76e1411ce20bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.720489501953125]], [[7.950204849243164]], [[8.676801681518555]], [[8.149105072021484]], [[7.102902412414551]], [[7.874119758605957]], [[8.240650177001953]], [[7.847740173339844]], [[7.982648849487305]], [[7.491360664367676]], [[8.735313415527344]], [[7.510238170623779]], [[8.066055297851562]], [[8.007878303527832]], [[8.85459041595459]], [[8.496610641479492]], [[8.094125747680664]], [[7.891778945922852]], [[8.22463607788086]], [[7.96538782119751]], [[7.587237358093262]], [[8.23629379272461]], [[8.250158309936523]], [[8.19361400604248]], [[8.194456100463867]], [[9.199542045593262]], [[7.542173385620117]], [[8.364714622497559]], [[7.895327091217041]], [[8.275627136230469]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_b5b16d26e52c291dbc628fe7ba3e2eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7b1e79dbb184cbc81782e5624757d25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.00590991973877]], [[7.415954113006592]], [[7.876989841461182]], [[6.942861080169678]], [[8.300236701965332]], [[7.491955757141113]], [[7.546222686767578]], [[7.608494758605957]], [[7.794553279876709]], [[7.298939228057861]], [[8.410886764526367]], [[7.41102409362793]], [[8.027511596679688]], [[7.821455478668213]], [[8.39918041229248]], [[8.598889350891113]], [[8.43173885345459]], [[8.188957214355469]], [[8.038849830627441]], [[7.986676216125488]], [[7.033593654632568]], [[7.4206156730651855]], [[8.053648948669434]], [[7.339644908905029]], [[7.243391513824463]], [[8.141550064086914]], [[7.59982967376709]], [[8.755999565124512]], [[8.071479797363281]], [[7.06417179107666]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_0b06757a9e08a064f96c2676e7acbed1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de1bde39c2e857057eb7aed401e3922a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba576fe25f7405e123f792d71e5cb42a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.14487075805664]], [[8.147989273071289]], [[8.095406532287598]], [[7.036656856536865]], [[7.791210174560547]], [[7.92747688293457]], [[7.291300296783447]], [[7.665894031524658]], [[8.080425262451172]], [[7.467463970184326]], [[7.3757219314575195]], [[7.0921525955200195]], [[6.8974761962890625]], [[7.289829730987549]], [[7.877655982971191]], [[7.310974597930908]], [[7.013061046600342]], [[6.738163948059082]], [[7.917231559753418]], [[6.857556343078613]], [[8.298291206359863]], [[7.9575042724609375]], [[7.84797477722168]], [[8.23776912689209]], [[7.439065933227539]], [[8.512922286987305]], [[7.981821060180664]], [[7.359304428100586]], [[6.39490270614624]], [[7.269107818603516]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_476204a4f959e2000bc428a5432e97b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0138115882873535]], [[3.2304744720458984]], [[3.1001744270324707]], [[3.620997667312622]], [[3.425168514251709]], [[3.1345767974853516]], [[3.0422685146331787]], [[3.525670289993286]], [[3.386475086212158]], [[3.072099447250366]], [[3.4267210960388184]], [[2.8959836959838867]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_3dad523efb5d861c0f1afe57db97e64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.5646629333496094]], [[3.4088218212127686]], [[4.021111011505127]], [[2.847135305404663]], [[3.5908870697021484]], [[2.934852361679077]], [[3.508193016052246]], [[3.6472582817077637]], [[3.661288261413574]], [[3.0511858463287354]], [[3.5024564266204834]], [[3.0943455696105957]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_29e3b48f101d1649c4f7b95710cbbcca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.729383945465088]], [[7.137293338775635]], [[6.083255290985107]], [[6.352455139160156]], [[7.294790267944336]], [[7.118271827697754]], [[7.140522003173828]], [[7.952057838439941]], [[7.3802666664123535]], [[7.0913896560668945]], [[6.739432334899902]], [[7.589100360870361]], [[7.090548515319824]], [[6.952294826507568]], [[7.234918594360352]], [[7.350093364715576]], [[6.242755889892578]], [[6.655361175537109]], [[5.618881702423096]], [[7.279906272888184]], [[6.899659633636475]], [[7.120492935180664]], [[7.67257022857666]], [[7.347502708435059]], [[6.531886100769043]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_8602fbad389cc2e19caacb74f22a685a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81d2487d849229bb76b6f691d0ada378(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9bd6738d2c5b774f4f6f52e3349f8de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f97303ef1727fe39f67cb99b38b8d3b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_204f9cd3793a715d8a64bcccd21fa547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5faf731ccc095c4ae364b5d7384d5d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.62026834487915]], [[4.912600517272949]], [[4.501530170440674]], [[5.113858699798584]], [[5.229439735412598]], [[4.599358081817627]], [[5.316549777984619]], [[4.630554676055908]], [[4.967809200286865]], [[4.93235969543457]], [[5.05343770980835]], [[5.221227169036865]], [[4.371150493621826]], [[5.340354919433594]], [[5.351968288421631]], [[5.038957118988037]], [[4.813811779022217]], [[5.55605411529541]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_6480f898595034dc31697d642337f1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_375b967a3fc3d07e21c648822c8cdf6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3406821489334106]], [[1.40310800075531]], [[1.3875422477722168]], [[1.3541191816329956]], [[1.5349763631820679]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_db3962e468602d8e444358b494617d45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.891082525253296]], [[2.8003101348876953]], [[3.0362467765808105]], [[3.3644373416900635]], [[2.7299394607543945]], [[3.5558624267578125]], [[3.410123109817505]], [[3.1943938732147217]], [[3.5989999771118164]], [[2.9150657653808594]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_4b0cd9ee28d8e1ad418a8cfb7ac041e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.823153972625732]], [[5.129012584686279]], [[5.525485515594482]], [[4.958107948303223]], [[5.484551906585693]], [[4.751092910766602]], [[5.324677467346191]], [[6.043829917907715]], [[5.8095316886901855]], [[5.279562473297119]], [[5.300936222076416]], [[6.204381465911865]], [[5.513743877410889]], [[5.982616424560547]], [[5.167477607727051]], [[5.536433219909668]], [[4.966822147369385]], [[5.783491134643555]], [[4.380859851837158]], [[5.523407459259033]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b06757a9e08a064f96c2676e7acbed1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd59a66ddaffc92e4cd9b0eaddbd4565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7d5c9ba710769629ee23b3989586a25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bec7913f453fadfede8a72c7c7e2766(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.116128921508789]], [[6.4880290031433105]], [[6.603219032287598]], [[5.883945465087891]], [[5.97711706161499]], [[6.399841785430908]], [[6.568505764007568]], [[7.100306987762451]], [[6.596995830535889]], [[6.940871238708496]], [[6.557913780212402]], [[6.544834613800049]], [[5.847147464752197]], [[7.082651138305664]], [[6.433468341827393]], [[6.7771406173706055]], [[6.6440534591674805]], [[6.463765621185303]], [[6.267218112945557]], [[6.189576625823975]], [[7.43379020690918]], [[6.055135250091553]], [[6.613389492034912]], [[6.876086711883545]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_2fe93505efa16cf7c1067f074d59de7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c91a7002a4880d3d6cbc19fbf53c91f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0780367851257324]], [[2.8307974338531494]], [[2.5395758152008057]], [[2.5772764682769775]], [[3.0108275413513184]], [[3.0733754634857178]], [[2.7212250232696533]], [[3.425830125808716]], [[3.5829856395721436]], [[2.8356549739837646]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_c5aa3cfcb25406ed9f2b28d40e508dcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24ec5768aef24a8ff8ec0aaa20ff87c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1a66d3e23eb30b294ab23bee69358eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e429cc0f89176fed69c77349906b447(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4abdeddc1141f15783e9cc19fdfcc025(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.2628655433654785]], [[5.365011692047119]], [[5.038057804107666]], [[5.306423187255859]], [[5.208133697509766]], [[4.970277786254883]], [[5.335209369659424]], [[5.289772033691406]], [[4.597698211669922]], [[4.871645450592041]], [[4.971390724182129]], [[4.5013556480407715]], [[5.485369682312012]], [[4.997920036315918]], [[5.272904872894287]], [[4.238986968994141]], [[5.5695109367370605]], [[4.83064079284668]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_b49625e8eabe189950ee47ce0414b92d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[8.870004653930664, 9.070971488952637, 8.6998291015625, 8.771418571472168, 8.77302074432373, 8.278258323669434, 9.320741653442383, 8.51421070098877, 8.791091918945312, 8.457852363586426, 9.51916790008545, 8.142207145690918, 8.19686508178711, 8.17872428894043, 8.666762351989746, 8.191591262817383, 9.497175216674805, 8.180706024169922, 8.71663761138916, 9.759553909301758, 8.672922134399414, 8.273152351379395, 8.350447654724121, 9.068215370178223, 8.876349449157715, 8.625724792480469, 9.934979438781738, 8.560579299926758, 8.065917015075684, 9.131192207336426]], dtype='float32').reshape([1, 30]),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973e03ebee097ba10130598c59d9fa41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028a9eef76aedb2b56b27c7546e7529(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.780267715454102]], [[8.131998062133789]], [[8.939107894897461]], [[8.882832527160645]], [[9.453350067138672]], [[8.85619068145752]], [[8.147751808166504]], [[8.396537780761719]], [[8.775067329406738]], [[8.127144813537598]], [[8.213729858398438]], [[8.807175636291504]], [[8.307685852050781]], [[9.488317489624023]], [[7.320240497589111]], [[8.850549697875977]], [[8.130460739135742]], [[8.228346824645996]], [[8.956204414367676]], [[8.420777320861816]], [[8.549696922302246]], [[7.867123126983643]], [[8.182889938354492]], [[8.662212371826172]], [[7.8356781005859375]], [[8.064471244812012]], [[7.9256134033203125]], [[7.720449447631836]], [[9.509540557861328]], [[8.814489364624023]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_5e7936974db74087915b595b064e70db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.213860273361206]], [[0.7169217467308044]], [[1.3732637166976929]], [[0.9356551766395569]], [[0.9903314113616943]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_b030a18e290a0e3e22599567d10a8143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.469463348388672]], [[3.039804458618164]], [[3.2300972938537598]], [[2.7001450061798096]], [[2.6241466999053955]], [[2.543884515762329]], [[2.668975591659546]], [[1.845589518547058]], [[2.947535514831543]], [[2.841306686401367]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_9b1898565c7cf24a012337b25dc8b1b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.91616153717041]], [[4.971972942352295]], [[4.818826675415039]], [[4.943068027496338]], [[4.595055103302002]], [[5.193109035491943]], [[4.6342620849609375]], [[4.798951625823975]], [[5.12747859954834]], [[4.536078929901123]], [[3.727442502975464]], [[4.675539016723633]], [[5.137940406799316]], [[5.504049777984619]], [[4.462343692779541]], [[4.524420738220215]], [[5.013905048370361]], [[5.048361778259277]], [[4.960768222808838]], [[5.547129154205322]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9efa920452a5502c02db3f4acdda5d46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.84181809425354]], [[3.919459342956543]], [[3.943265914916992]], [[3.615178108215332]], [[4.343533515930176]], [[4.357351303100586]], [[3.5953638553619385]], [[4.600090503692627]], [[3.89821195602417]], [[4.208963871002197]], [[4.014341831207275]], [[4.141739368438721]], [[4.350880146026611]], [[4.032442569732666]], [[4.039931774139404]], [[4.096301555633545]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_4ad8e88dfbe306404163a497ebe4e71a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6806d448331b5db78b1c9870b5815104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fda5bcd75ed21a468adca6f70c606be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605a7a326b2d5016653b8e5fcc39577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95f7aba09dfa233495c1c2c117dff927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ce4909a7eb52aa0abd98f23838ca0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3965838c12f97689d74e974b95e372d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e341740e0f212bca7cbc0355e70c8765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d3adbfd2eadaa6a75d16f583fd2691b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fb2df61e7ce2bb84740ce8bc25e1792(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.8748505115509033]], [[3.848036527633667]], [[3.7055251598358154]], [[3.35248064994812]], [[2.893239974975586]], [[4.06261682510376]], [[3.7211146354675293]], [[4.051092624664307]], [[3.9451940059661865]], [[4.032584190368652]], [[4.147378921508789]], [[2.9157514572143555]], [[3.8111705780029297]], [[3.7591934204101562]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_00960b09ee123c4b813d2091c973fca6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.051861763000488]], [[5.754777431488037]], [[4.818517684936523]], [[4.752799034118652]], [[4.6054182052612305]], [[5.425618648529053]], [[5.631992340087891]], [[5.630918979644775]], [[5.270631313323975]], [[5.155634880065918]], [[5.055809020996094]], [[4.907738208770752]], [[5.484491348266602]], [[4.884391784667969]], [[5.892811298370361]], [[5.109838962554932]], [[5.606997489929199]], [[5.1103386878967285]], [[5.412006378173828]], [[4.9560112953186035]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_3592761e157a9316e8d14f8f68770b76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd59a66ddaffc92e4cd9b0eaddbd4565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efe5f730bcd742899e2e7008e8cc00f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.228400707244873]], [[7.572210311889648]], [[7.720287799835205]], [[7.640965461730957]], [[6.785443305969238]], [[8.622623443603516]], [[8.482710838317871]], [[7.288590431213379]], [[7.2805681228637695]], [[8.149312973022461]], [[7.05112361907959]], [[8.065946578979492]], [[7.8982014656066895]], [[8.205387115478516]], [[8.06624698638916]], [[7.016504287719727]], [[8.207212448120117]], [[6.216163158416748]], [[7.654832363128662]], [[7.563879013061523]], [[7.733938217163086]], [[7.262387275695801]], [[7.852588176727295]], [[7.543629169464111]], [[8.282045364379883]], [[7.994214057922363]], [[7.4732866287231445]], [[8.085190773010254]], [[8.281085968017578]], [[7.778512477874756]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6806d448331b5db78b1c9870b5815104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8c794a3a44e5cb2bdf149bff9a192f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e493737f3bb89a99d58681fed0f43fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0865c6a08f6f43b3e1abc77446daaffa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0865c6a08f6f43b3e1abc77446daaffa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e493737f3bb89a99d58681fed0f43fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0865c6a08f6f43b3e1abc77446daaffa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0865c6a08f6f43b3e1abc77446daaffa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39439121ebb1684dbb112caa95efaf7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_905f07b249425e820409872a936fce77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_905f07b249425e820409872a936fce77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36d761e5755aeeb49fd5bf4086dc6072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de41b9adcaa6cd010c453b5f093026ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de41b9adcaa6cd010c453b5f093026ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9738e3afb791bd248e4209b9daa7d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1f44b8f22d8150faf13ac5fe97aa2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1f44b8f22d8150faf13ac5fe97aa2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9738e3afb791bd248e4209b9daa7d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1f44b8f22d8150faf13ac5fe97aa2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1f44b8f22d8150faf13ac5fe97aa2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a32fb47d802b8ea17125a4f9e682c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea0464efeaaa9086a475c84ea89e5b79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea0464efeaaa9086a475c84ea89e5b79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5656a433af1c84439624bf8011e740d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_789064b29d36c8dcee144fd2d4f56e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_789064b29d36c8dcee144fd2d4f56e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaf63112a9a83b8e558fd0f60e925e02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b06757a9e08a064f96c2676e7acbed1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c247d1ada10f78b0c70d7a8632039ed8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04845b1dab159278b9ff8ce55f81ddfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3a799135057b9f2577663f9b1710b4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.9753828048706055]], [[6.003129005432129]], [[7.127122402191162]], [[5.840324401855469]], [[6.891082763671875]], [[6.126930236816406]], [[7.248678684234619]], [[6.783142566680908]], [[6.2587175369262695]], [[5.727454662322998]], [[7.110597133636475]], [[6.120436668395996]], [[7.036701202392578]], [[6.22518253326416]], [[7.063243389129639]], [[6.6329193115234375]], [[6.64615535736084]], [[5.935981750488281]], [[6.5288004875183105]], [[7.519911289215088]], [[6.803238391876221]], [[6.365416526794434]], [[5.566114902496338]], [[7.818721294403076]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_e5dd5e3d7591c9909289318d1feac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.455387115478516]], [[6.70116662979126]], [[6.931902885437012]], [[6.813783168792725]], [[7.750301361083984]], [[7.221365451812744]], [[6.216297626495361]], [[6.400246620178223]], [[7.3066630363464355]], [[7.40735387802124]], [[6.200910568237305]], [[7.3968400955200195]], [[6.916532039642334]], [[7.136009693145752]], [[7.048123359680176]], [[7.445611953735352]], [[6.951174259185791]], [[6.86937952041626]], [[7.565418243408203]], [[6.618014335632324]], [[7.1575422286987305]], [[6.011700630187988]], [[7.5044941902160645]], [[6.530730247497559]], [[6.473878383636475]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_453e4cd53446c7d72fc1b0c6f1e75cea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.961212396621704]], [[2.8444416522979736]], [[3.0302486419677734]], [[3.4965062141418457]], [[4.392430305480957]], [[3.4449360370635986]], [[4.060918807983398]], [[2.69504451751709]], [[3.8072237968444824]], [[3.5579631328582764]], [[3.595301628112793]], [[3.4865787029266357]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd59a66ddaffc92e4cd9b0eaddbd4565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de809e96744bf133127ff3200ee3b0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5c58e4748f80a4defc5ffd856ba1830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ad8e88dfbe306404163a497ebe4e71a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f152e22495fde7023ec84609be7270b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9147612ecfabda705c703b917857ca82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9e97999e791770d9bf8126e24b11f37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e103ed3058231c8dfadb7ee1a54e9e18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[706.5451049804688]], [[698.1089477539062]], [[733.5790405273438]], [[774.1317138671875]], [[702.3773803710938]], [[767.1760864257812]], [[719.3434448242188]], [[709.82080078125]], [[691.0403442382812]], [[681.920166015625]], [[771.4940795898438]], [[669.5833740234375]], [[718.261962890625]], [[709.01171875]], [[721.8114013671875]], [[735.890625]], [[706.6136474609375]], [[698.2988891601562]], [[718.6580200195312]], [[772.4765014648438]], [[658.5473022460938]], [[714.2171630859375]], [[651.621337890625]], [[754.21337890625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_6398f9a539d15af712b3e2463ed89dad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[70.37177276611328]], [[75.7459487915039]], [[73.71358489990234]], [[74.7445068359375]], [[72.09110260009766]], [[75.07721710205078]], [[75.77609252929688]], [[74.45829010009766]], [[70.37089538574219]], [[78.90221405029297]], [[83.17375946044922]], [[70.34793090820312]], [[81.13215637207031]], [[73.42264556884766]], [[70.3205795288086]], [[73.20497131347656]], [[77.08792114257812]], [[72.10660552978516]], [[67.34578704833984]], [[75.58836364746094]], [[74.80601501464844]], [[75.42642211914062]], [[81.91889953613281]], [[68.68610382080078]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_a14655904aa6384c9b05f0a3698d7ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45.29734420776367]], [[51.2456169128418]], [[43.81867218017578]], [[44.45344924926758]], [[44.131744384765625]], [[45.768131256103516]], [[46.679420471191406]], [[42.54667663574219]], [[38.03548812866211]], [[47.30936813354492]], [[46.49201202392578]], [[46.001251220703125]], [[39.304664611816406]], [[44.54432678222656]], [[44.52653121948242]], [[44.796451568603516]], [[49.56315231323242]], [[45.860416412353516]], [[42.550559997558594]], [[45.2556266784668]], [[46.87994384765625]], [[43.06817626953125]], [[48.176002502441406]], [[40.50491714477539]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_c4aa506f8d8b394d907db96b25d82709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[22.481063842773438]], [[22.24987030029297]], [[22.205080032348633]], [[23.617107391357422]], [[22.71398162841797]], [[22.377277374267578]], [[23.214012145996094]], [[21.080718994140625]], [[21.404701232910156]], [[20.12889289855957]], [[23.075468063354492]], [[21.405946731567383]], [[23.80116081237793]], [[23.009891510009766]], [[20.116384506225586]], [[22.51435661315918]], [[22.5460205078125]], [[23.645549774169922]], [[22.766191482543945]], [[21.108200073242188]], [[21.514690399169922]], [[21.464120864868164]], [[23.93003273010254]], [[21.577220916748047]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_b2cdb2b58e45d3ef1d5b0142cbd1344a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33464.0625]], [[34128.3203125]], [[34373.05859375]], [[28562.703125]], [[32737.2890625]], [[36944.26953125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_880eab8e3dee57628ac94c07ed6121cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[40044.80078125]], [[37509.421875]], [[43190.484375]], [[37027.70703125]], [[37379.66015625]], [[33082.0859375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_0837ff447d46896bacec7af88a2c0725(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45157.8984375]], [[35525.29296875]], [[38019.31640625]], [[36812.3671875]], [[32309.341796875]], [[43327.96484375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_b533f4b0cb2717cfd9f33f9326b2558f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[46890.375]], [[40730.390625]], [[41829.359375]], [[43960.64453125]], [[42763.0546875]], [[40935.3671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_620c5f06609ead8a618442180f90bbc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8602fbad389cc2e19caacb74f22a685a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3b5927c9b91ed3a82f36305cfae72f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_494d14d07a4909167cfbb60b464ec431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.723304271697998]], [[6.127195835113525]], [[5.915198802947998]], [[5.47707986831665]], [[5.763934135437012]], [[5.835768699645996]], [[6.205861568450928]], [[5.506523132324219]], [[5.576387405395508]], [[5.98674201965332]], [[5.432145595550537]], [[5.712860584259033]], [[5.408754825592041]], [[5.702744960784912]], [[6.343686103820801]], [[6.0185065269470215]], [[5.901707172393799]], [[5.767567157745361]], [[5.454860687255859]], [[5.4875054359436035]], [[6.7340288162231445]], [[6.339094161987305]], [[6.3797926902771]], [[5.385677337646484]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_a103e83c2b09e2dad0002fe9d31bc6fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18c96088ac086d8e176b8ecfdf7816a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fa3b08455f8eb2cf5d773e41c00f54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()