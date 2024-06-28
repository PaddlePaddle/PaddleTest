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


    class TestPrimitiveOp_eac6ae602e2ad58e40abefa1d3369ea8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202bfdc0ac9db07356cae4b693623ccc
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.33536958694458, 5.728888511657715, 4.716067790985107, 4.229830265045166, 5.404992580413818, 4.498203754425049, 5.148319721221924, 4.750048637390137, 5.1595282554626465, 4.218147277832031, 5.205923080444336, 4.9782233238220215, 4.736413478851318, 4.962367057800293, 5.054198265075684, 5.145442008972168, 4.78067684173584, 4.842703819274902]], dtype='float32').reshape([1, 18]),
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


    class TestPrimitiveOp_2a4f918bb44c9189f531e7cd185ae8b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd6460d9338b9f965d298e71d4ef198a
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.162670612335205, 4.743340492248535, 5.998516082763672, 5.040639877319336, 5.3741350173950195, 5.7692341804504395, 5.374410152435303, 5.690393447875977, 6.033936023712158, 5.206454277038574, 5.5260396003723145, 5.6504645347595215, 5.447342395782471, 5.529510974884033, 5.592252254486084, 6.441046714782715, 5.761960983276367, 5.791680812835693, 5.865000247955322, 5.504603385925293, 5.820616245269775, 5.220635890960693, 5.280458450317383]], dtype='float32').reshape([1, 23]),
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


    class TestPrimitiveOp_e978caf29c571d5e4a0827027a5e73c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.156549453735352]], [[6.578279972076416]], [[8.965864181518555]], [[7.614697456359863]], [[7.707426071166992]], [[7.436953067779541]], [[7.509988307952881]], [[7.788017749786377]], [[6.393144607543945]], [[7.362917900085449]], [[7.2338972091674805]], [[8.074005126953125]], [[7.098735809326172]], [[6.910137176513672]], [[8.077880859375]], [[8.106316566467285]], [[7.840653419494629]], [[8.020371437072754]], [[8.272087097167969]], [[8.352813720703125]], [[7.642253875732422]], [[7.687630653381348]], [[7.613253593444824]], [[7.520717620849609]], [[7.637831687927246]], [[8.50828742980957]], [[6.867826461791992]], [[7.150737762451172]], [[7.4321746826171875]], [[7.626999378204346]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_391c3a6fc28b803c00e77b3a4d2c40eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.764479637145996]], [[8.717795372009277]], [[8.243857383728027]], [[6.911282062530518]], [[7.926513671875]], [[8.030611991882324]], [[8.945138931274414]], [[7.763011932373047]], [[7.695864200592041]], [[7.878115177154541]], [[8.235343933105469]], [[6.995920658111572]], [[8.233352661132812]], [[7.727132797241211]], [[7.220941066741943]], [[8.071619033813477]], [[8.144110679626465]], [[6.693134307861328]], [[7.199584484100342]], [[8.235687255859375]], [[7.900215148925781]], [[7.703789234161377]], [[7.636146545410156]], [[8.11130428314209]], [[8.535118103027344]], [[7.386641502380371]], [[8.645404815673828]], [[6.932118892669678]], [[7.863061904907227]], [[8.122811317443848]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_c1ee1a7be73ff6be98239f2649f61a22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa6a208551763b029a4175fcd015eae
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8566051721572876]], [[2.095454216003418]], [[1.8238812685012817]], [[1.474919080734253]], [[1.5276159048080444]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


    class TestPrimitiveOp_e3ea30020a1a5d29ab95c8128292ca05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.719106674194336]], [[2.747519016265869]], [[2.2771918773651123]], [[2.710972547531128]], [[2.6780829429626465]], [[2.6255321502685547]], [[2.8236663341522217]], [[2.8805809020996094]], [[2.647463321685791]], [[2.6628236770629883]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_93bc79a537524d99d3fae3613c230e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.086631774902344]], [[7.308384895324707]], [[5.830771446228027]], [[6.009194850921631]], [[5.87945032119751]], [[6.114314079284668]], [[5.7845635414123535]], [[6.975741386413574]], [[5.626193523406982]], [[6.476055145263672]], [[6.509440898895264]], [[6.814225673675537]], [[6.179939270019531]], [[6.212969779968262]], [[5.426244735717773]], [[5.825172424316406]], [[6.444512367248535]], [[6.775128364562988]], [[5.420651435852051]], [[6.370400428771973]], [[6.124203205108643]], [[6.66068172454834]], [[6.493947505950928]], [[6.2271623611450195]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_ed1a1db49ab27adc2bd3b1d5d96ca9dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.429371356964111]], [[5.102461814880371]], [[5.122546195983887]], [[4.959865093231201]], [[4.487847328186035]], [[4.595395565032959]], [[4.401482582092285]], [[5.045962333679199]], [[5.277590274810791]], [[4.407054424285889]], [[4.078505992889404]], [[4.891146183013916]], [[4.863668918609619]], [[4.882938385009766]], [[4.876100540161133]], [[4.58686637878418]], [[4.975370407104492]], [[5.126645565032959]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fff9f5f118a14115c6363026a24a7cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.2779717445373535]], [[6.342955589294434]], [[6.889688014984131]], [[6.494229316711426]], [[6.744277000427246]], [[6.069737911224365]], [[7.056484222412109]], [[5.741796493530273]], [[6.317704677581787]], [[6.743081569671631]], [[6.25610876083374]], [[6.39182186126709]], [[5.538364410400391]], [[6.538943290710449]], [[6.79679536819458]], [[7.517889022827148]], [[6.390634536743164]], [[6.382289886474609]], [[6.212860584259033]], [[6.586852073669434]], [[5.402479648590088]], [[6.590733051300049]], [[6.075348854064941]], [[5.696362495422363]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_d355fdde4daa3a39374268d59fe0d41b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81236129c333dfe7ae73bbcbb0979cbf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5981800556182861]], [[1.4517943859100342]], [[0.8325352668762207]], [[1.1580177545547485]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_b4ff93a5e5c21e6087c2defcc7ce3eae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a79f19dffaf1a401b1a360fa95eb71
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.759239673614502]], [[3.067654848098755]], [[2.8620004653930664]], [[2.730835437774658]], [[2.2491555213928223]], [[2.969370126724243]], [[3.7615504264831543]], [[2.9485251903533936]], [[2.366492748260498]], [[3.176340103149414]], [[2.972701072692871]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_5eecc341fd6d71eb45204605b8c0ad33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[9.219853401184082]], [[7.977720737457275]], [[7.273802280426025]], [[8.22716236114502]], [[8.4916410446167]], [[9.26848316192627]], [[7.195425987243652]], [[7.818164825439453]], [[8.419690132141113]], [[7.483343601226807]], [[8.165380477905273]], [[7.699680328369141]], [[8.12510871887207]], [[8.931337356567383]], [[7.405519485473633]], [[9.185028076171875]], [[7.8361005783081055]], [[7.829963684082031]], [[8.57872200012207]], [[7.620147705078125]], [[8.942919731140137]], [[7.511859893798828]], [[7.994983196258545]], [[7.808498859405518]], [[8.88344955444336]], [[7.918191432952881]], [[8.144401550292969]], [[8.468025207519531]], [[8.549529075622559]], [[8.860572814941406]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_8c25cc36ea1f2bb1a0b363e4224b9e9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.339223384857178]], [[4.631780624389648]], [[4.155935764312744]], [[3.9742817878723145]], [[5.302306652069092]], [[4.456806659698486]], [[4.654520034790039]], [[4.203819751739502]], [[4.233545303344727]], [[3.887152910232544]], [[4.019169807434082]], [[4.785059452056885]], [[5.198037147521973]], [[4.3127031326293945]], [[3.808553695678711]], [[4.056495666503906]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_424e52a9c9a34fb01df7e7900ba69883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.9523186683654785]], [[7.42242431640625]], [[8.545364379882812]], [[8.203980445861816]], [[7.4313483238220215]], [[7.6026763916015625]], [[8.048667907714844]], [[8.16807746887207]], [[7.477578163146973]], [[7.682559967041016]], [[7.5439887046813965]], [[8.519378662109375]], [[8.315771102905273]], [[7.916327476501465]], [[7.257624626159668]], [[8.291592597961426]], [[7.157421588897705]], [[7.265719890594482]], [[7.141048908233643]], [[7.891236305236816]], [[8.022711753845215]], [[8.24140453338623]], [[7.856504917144775]], [[8.895251274108887]], [[8.04963493347168]], [[8.79507064819336]], [[8.328497886657715]], [[7.987967491149902]], [[8.476158142089844]], [[7.17471170425415]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_8c28c2cc9d0ff2d9d801a42b61b4bd85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.717169761657715]], [[6.377462387084961]], [[6.11559534072876]], [[6.248219966888428]], [[6.040029048919678]], [[7.45786714553833]], [[7.106032371520996]], [[5.864388465881348]], [[6.851099014282227]], [[6.623362064361572]], [[6.710056781768799]], [[5.612614154815674]], [[7.330811023712158]], [[5.56170129776001]], [[7.341120719909668]], [[6.5250678062438965]], [[6.4778265953063965]], [[6.345496654510498]], [[7.3401031494140625]], [[6.775630474090576]], [[6.1982197761535645]], [[5.557824611663818]], [[6.792047023773193]], [[6.723367214202881]], [[6.0354838371276855]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_ca6506fb7d8bcd811ad436b1d5eb2697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.431391716003418]], [[4.728453636169434]], [[3.8485469818115234]], [[3.7370498180389404]], [[4.240548133850098]], [[3.7491250038146973]], [[4.5072808265686035]], [[4.023393154144287]], [[3.9357423782348633]], [[4.149059295654297]], [[4.889106273651123]], [[4.266361236572266]], [[4.887792587280273]], [[4.657820701599121]], [[4.33989143371582]], [[4.628081321716309]], [[3.937661647796631]], [[4.499518394470215]], [[4.870829105377197]], [[4.252282619476318]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_4710dc1c1f05b68cc701f83ac4a56f30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.285511016845703]], [[4.616420745849609]], [[5.1114630699157715]], [[4.157078742980957]], [[4.800133228302002]], [[3.777223587036133]], [[4.694716930389404]], [[3.983590841293335]], [[3.9912760257720947]], [[4.235405921936035]], [[4.37108039855957]], [[4.672853469848633]], [[4.280800819396973]], [[4.807054042816162]], [[4.614780426025391]], [[4.6422810554504395]], [[4.211650371551514]], [[5.195508003234863]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_c9bc6f86abb81ad7183f4b52c079a74b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.78338623046875]], [[4.758307456970215]], [[4.5570759773254395]], [[4.651852607727051]], [[4.396996021270752]], [[4.2491278648376465]], [[4.278838634490967]], [[4.917035102844238]], [[4.2762885093688965]], [[4.933725833892822]], [[5.054064750671387]], [[4.185392379760742]], [[4.766103267669678]], [[4.256556510925293]], [[4.734835147857666]], [[5.001825332641602]], [[3.9590556621551514]], [[4.636456489562988]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_9f23e5f65f2d81fcc0b69e8c24dfc3fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f94098bffcb0a586cff63c780c80b01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.475754737854004]], [[5.490368366241455]], [[5.066490650177002]], [[6.535336494445801]], [[6.574573993682861]], [[6.074638843536377]], [[5.812880039215088]], [[6.25173282623291]], [[6.593845367431641]], [[6.095102787017822]], [[6.271059036254883]], [[5.55356502532959]], [[6.959198951721191]], [[5.8360443115234375]], [[6.394191741943359]], [[5.082563400268555]], [[5.476058006286621]], [[5.893261909484863]], [[5.3520402908325195]], [[6.88227653503418]], [[6.345145225524902]], [[5.516343116760254]], [[6.1561150550842285]], [[6.437520980834961]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_6a4c5bfd3d6afc713c0678b8f95a670a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e89faec298d465a2daee39fefe711bd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.772536277770996]], [[5.173009395599365]], [[3.8997905254364014]], [[5.0453996658325195]], [[4.3154191970825195]], [[5.118714809417725]], [[4.704500198364258]], [[4.647329807281494]], [[4.215498924255371]], [[4.829869270324707]], [[4.632633686065674]], [[5.515543460845947]], [[4.459994792938232]], [[4.442266464233398]], [[4.744085788726807]], [[4.792042255401611]], [[4.764348030090332]], [[4.661513805389404]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_e0769dd5fb01ede27c29f9cb69b14a3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.520150184631348]], [[4.817781925201416]], [[4.683832168579102]], [[5.0453901290893555]], [[4.88814115524292]], [[3.796255111694336]], [[5.0784525871276855]], [[4.307088375091553]], [[4.392723560333252]], [[5.034638404846191]], [[4.890681266784668]], [[4.789068222045898]], [[4.620427131652832]], [[4.711824893951416]], [[4.296992778778076]], [[4.6062116622924805]], [[4.63204288482666]], [[5.542410373687744]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_30b1115e81f3cac24d0fe6a4e0184e5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.531711101531982]], [[4.344454288482666]], [[4.5150837898254395]], [[4.630534648895264]], [[4.285594463348389]], [[4.032207489013672]], [[3.536329746246338]], [[4.6597137451171875]], [[4.211676597595215]], [[4.191618919372559]], [[4.450105667114258]], [[5.07877779006958]], [[4.409242153167725]], [[4.230588436126709]], [[4.868673324584961]], [[4.5204758644104]], [[5.747686386108398]], [[4.947834491729736]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_53eb6f8cc385506ccfa288d6d1e8349a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.25018835067749]], [[4.311614513397217]], [[4.094573497772217]], [[3.818601131439209]], [[3.6546530723571777]], [[3.475036144256592]], [[3.1861672401428223]], [[3.8367486000061035]], [[4.038932800292969]], [[4.63492488861084]], [[3.6220250129699707]], [[3.9613070487976074]], [[3.4617204666137695]], [[4.631092548370361]], [[3.767266273498535]], [[4.41674280166626]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_a3771827d567bf68efe8e53ee6f221a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.675265789031982]], [[4.467334270477295]], [[3.612835645675659]], [[4.771159648895264]], [[4.658884525299072]], [[4.322079658508301]], [[4.130802631378174]], [[4.945823669433594]], [[4.1770548820495605]], [[4.741894245147705]], [[4.489341735839844]], [[4.604240417480469]], [[4.373265266418457]], [[3.9902279376983643]], [[4.216479778289795]], [[4.061892986297607]], [[4.76384973526001]], [[3.904416561126709]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_cf440ada98ccaf23f365cfd58cffc53c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81236129c333dfe7ae73bbcbb0979cbf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9944493174552917]], [[1.2166593074798584]], [[1.1984256505966187]], [[1.1523396968841553]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_d95feb2314db028a705a469fbea0340d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.8062310218811035]], [[5.361790180206299]], [[4.5029802322387695]], [[4.500344753265381]], [[4.982926845550537]], [[4.848570346832275]], [[4.778101921081543]], [[5.076159954071045]], [[4.993417263031006]], [[4.963909149169922]], [[4.629103660583496]], [[4.729391098022461]], [[5.710188865661621]], [[4.812963008880615]], [[5.511858940124512]], [[5.036686420440674]], [[5.425128936767578]], [[4.4527363777160645]], [[5.328535556793213]], [[5.170138359069824]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_245a6930c8f82ef5007bae20918d043e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.342505693435669]], [[2.7520391941070557]], [[2.714287519454956]], [[3.211639881134033]], [[3.0043015480041504]], [[3.492180347442627]], [[3.593163013458252]], [[3.679666042327881]], [[2.5081582069396973]], [[3.2115612030029297]], [[3.288628101348877]], [[3.0804831981658936]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_5203c0681fa7afc74ef2f916e0f8aeb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.7912914752960205]], [[4.724730014801025]], [[5.298625946044922]], [[5.1255106925964355]], [[4.62327241897583]], [[5.113589286804199]], [[5.026547431945801]], [[4.239257335662842]], [[4.4622602462768555]], [[5.033857822418213]], [[3.9976367950439453]], [[4.5428996086120605]], [[4.076603889465332]], [[4.696175575256348]], [[4.704270839691162]], [[4.375837326049805]], [[4.985689640045166]], [[4.377310276031494]], [[4.806520938873291]], [[4.063906669616699]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_ef4553b206a5464ada6c0d3f29a19ae6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a79f19dffaf1a401b1a360fa95eb71
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0060856342315674]], [[2.823049545288086]], [[2.855907440185547]], [[3.3607177734375]], [[3.1934914588928223]], [[3.269962787628174]], [[2.899298667907715]], [[3.100433111190796]], [[3.093851089477539]], [[2.913404703140259]], [[2.969477653503418]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_873f6c89a1efcce420c2e851d6a291ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa5522cc0fff55438ea3c29c97b2341b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.2398808002471924]], [[3.508150577545166]], [[2.8001081943511963]], [[3.5751781463623047]], [[2.6233019828796387]], [[3.1378047466278076]], [[3.546370506286621]], [[3.4879634380340576]], [[3.134951114654541]], [[3.1070199012756348]], [[3.4020214080810547]], [[2.941101312637329]], [[3.3202311992645264]], [[3.3293280601501465]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


    class TestPrimitiveOp_e0851d6d271e80daef5251b73d0a8042(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.995167255401611]], [[5.068471908569336]], [[5.232618808746338]], [[4.3133463859558105]], [[5.070679664611816]], [[4.943248271942139]], [[5.22681188583374]], [[4.9932475090026855]], [[5.181914806365967]], [[4.9883599281311035]], [[4.587845325469971]], [[4.951524257659912]], [[4.5863518714904785]], [[4.5127482414245605]], [[4.808051109313965]], [[5.10516881942749]], [[4.906620979309082]], [[4.632011413574219]], [[4.7547783851623535]], [[4.731569766998291]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_6b728aaf630107b5ef6e32afdd66577d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43951.48828125]], [[34423.453125]], [[34047.71484375]], [[29783.375]], [[37019.40234375]], [[38497.50390625]]], [[[43313.31640625]], [[33921.72265625]], [[33559.36328125]], [[29355.10546875]], [[36486.9140625]], [[37942.50390625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_52c9c468f3d2df94bdecb77fd580e547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37845.80859375]], [[32601.22265625]], [[41104.3828125]], [[33579.23828125]], [[41183.9921875]], [[40182.13671875]]], [[[36832.75390625]], [[31724.77734375]], [[40006.65625]], [[32682.271484375]], [[40081.921875]], [[39108.6875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_a9ba78676797dc7f1e6c67b70b6a947b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[47123.75]], [[41793.8828125]], [[49294.4765625]], [[44015.36328125]], [[40978.10546875]], [[35567.5703125]]], [[[45730.7265625]], [[40560.58203125]], [[47835.48828125]], [[42716.0703125]], [[39762.51953125]], [[34511.59375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_5e1af31197f4f55f7cf91b75472a7362(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37736.85546875]], [[39921.3515625]], [[44999.49609375]], [[36545.47265625]], [[40640.953125]], [[44243.4921875]]], [[[36904.62109375]], [[39050.265625]], [[44017.51171875]], [[35748.12890625]], [[39746.23828125]], [[43278.5859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


    class TestPrimitiveOp_818544ff76813b0aaa03baab21e6e7b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.301014423370361]], [[8.403518676757812]], [[8.03527545928955]], [[8.760984420776367]], [[8.188155174255371]], [[7.942190170288086]], [[7.466071128845215]], [[7.920556545257568]], [[8.333847045898438]], [[8.276917457580566]], [[6.924700736999512]], [[7.403339385986328]], [[7.732781410217285]], [[8.625016212463379]], [[6.764622688293457]], [[7.116641044616699]], [[8.37065601348877]], [[8.823826789855957]], [[8.26421070098877]], [[8.392292022705078]], [[6.971035480499268]], [[7.877248287200928]], [[8.675994873046875]], [[7.90565299987793]], [[8.166635513305664]], [[8.00434684753418]], [[7.3291497230529785]], [[7.97578239440918]], [[7.904312610626221]], [[8.364042282104492]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_0cb19ce2c84c505806f255469600b7b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.341287136077881]], [[7.299825191497803]], [[8.345277786254883]], [[7.708090782165527]], [[6.989670753479004]], [[7.035129070281982]], [[7.876548767089844]], [[7.641441822052002]], [[7.257607460021973]], [[7.430710792541504]], [[7.43454647064209]], [[7.339380741119385]], [[7.949814319610596]], [[7.445698261260986]], [[7.395953178405762]], [[7.797006130218506]], [[7.37500524520874]], [[7.733424186706543]], [[7.49075984954834]], [[7.240258693695068]], [[7.815331935882568]], [[8.243738174438477]], [[7.368047714233398]], [[7.8810553550720215]], [[8.4630126953125]], [[7.198576927185059]], [[6.862584114074707]], [[8.420339584350586]], [[7.592401504516602]], [[7.650505542755127]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_093ed3cdc8f4d1b21619a0d4aa800009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de874dd1a295c2f0f6de159ce1f8aff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.334691047668457]], [[6.553110122680664]], [[7.550365924835205]], [[7.175827980041504]], [[7.56691837310791]], [[7.009180545806885]], [[7.625830173492432]], [[6.939151763916016]], [[7.596673488616943]], [[6.650327682495117]], [[7.773587703704834]], [[7.105720043182373]], [[7.4982757568359375]], [[6.71769905090332]], [[7.4405083656311035]], [[7.593581199645996]], [[7.8551530838012695]], [[7.435949802398682]], [[8.307052612304688]], [[7.227053642272949]], [[7.727344036102295]], [[7.8308796882629395]], [[7.484035968780518]], [[6.645956516265869]], [[7.704050064086914]], [[7.600241184234619]], [[6.777804851531982]], [[7.476434230804443]], [[7.012086868286133]], [[7.311110496520996]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_9815a79ed6ca45df159b8a6a7cbc06f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.924079895019531]], [[7.3153605461120605]], [[7.995610237121582]], [[7.2530198097229]], [[7.813193321228027]], [[7.922425270080566]], [[7.419620990753174]], [[7.603722095489502]], [[7.701774597167969]], [[7.354045867919922]], [[7.252942085266113]], [[8.358896255493164]], [[7.384546756744385]], [[8.001121520996094]], [[7.920747756958008]], [[7.5409440994262695]], [[7.896615505218506]], [[8.030363082885742]], [[8.464024543762207]], [[8.035039901733398]], [[7.323920726776123]], [[7.297919750213623]], [[7.611123085021973]], [[7.6905646324157715]], [[6.916043281555176]], [[6.890603542327881]], [[7.488044261932373]], [[6.946822166442871]], [[7.933925151824951]], [[7.751521587371826]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_9513874f19124a06da9901708466311a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.44456148147583]], [[3.229776382446289]], [[3.124418258666992]], [[3.338461399078369]], [[3.577549457550049]], [[3.919827699661255]], [[3.9549551010131836]], [[3.856282949447632]], [[4.125796318054199]], [[3.8217051029205322]], [[2.8720881938934326]], [[3.986161947250366]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_d55c10184fae82d210ace1ca3783cee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.021803140640259]], [[2.8979740142822266]], [[2.6711370944976807]], [[3.363351345062256]], [[2.642312526702881]], [[3.2148635387420654]], [[3.0581884384155273]], [[3.5348715782165527]], [[3.4332950115203857]], [[3.799393653869629]], [[3.1188201904296875]], [[2.997915029525757]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_b951f4cf01094fbd92b3ae3f260a909c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.129674911499023]], [[6.374298095703125]], [[6.090598106384277]], [[6.553129196166992]], [[6.5286946296691895]], [[6.757339000701904]], [[6.135092735290527]], [[6.8264689445495605]], [[6.315954685211182]], [[7.21508264541626]], [[6.141348838806152]], [[5.7715253829956055]], [[6.627416610717773]], [[6.3487701416015625]], [[6.562192916870117]], [[6.3334736824035645]], [[6.540079116821289]], [[6.168745517730713]], [[6.445700645446777]], [[6.114558696746826]], [[5.674906253814697]], [[6.7818779945373535]], [[6.279696941375732]], [[6.37258243560791]], [[6.395266532897949]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_83bc32f921152632d39c35329bcef94f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.234936237335205]], [[4.312356948852539]], [[4.602973937988281]], [[4.165308952331543]], [[4.123189926147461]], [[4.576277732849121]], [[4.517114639282227]], [[4.990133285522461]], [[4.928924083709717]], [[4.565418720245361]], [[4.6072797775268555]], [[4.104726791381836]], [[4.53456974029541]], [[4.675724983215332]], [[5.090818405151367]], [[4.430437088012695]], [[4.960774898529053]], [[4.91608190536499]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_aed6197d24914caf579adb50cfacd127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.412215232849121]], [[1.8416154384613037]], [[1.87169349193573]], [[1.7094300985336304]], [[1.4866801500320435]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


    class TestPrimitiveOp_14351e0caccc92af5479539218975ddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6466522216796875]], [[3.1592297554016113]], [[2.6499240398406982]], [[3.1042873859405518]], [[2.4753024578094482]], [[2.788867950439453]], [[3.2296197414398193]], [[2.8998422622680664]], [[2.356828212738037]], [[2.5325305461883545]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_98e0f932698350d637eec48b8bab85c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.219168663024902]], [[4.386056900024414]], [[4.218252658843994]], [[3.8906869888305664]], [[3.821678876876831]], [[5.3758955001831055]], [[4.510594844818115]], [[4.741876125335693]], [[4.260367393493652]], [[3.830246686935425]], [[4.32592248916626]], [[4.207342147827148]], [[4.228832244873047]], [[4.594235420227051]], [[4.741360664367676]], [[4.85493803024292]], [[4.056390762329102]], [[4.101260185241699]], [[5.8075480461120605]], [[4.5099711418151855]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_bc3a8f8f3e94ad88b5db9b05907a084e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.851374626159668]], [[6.279842853546143]], [[6.162980556488037]], [[6.155672550201416]], [[6.570970058441162]], [[6.711134910583496]], [[6.90558385848999]], [[6.065819263458252]], [[7.283596992492676]], [[7.3350701332092285]], [[7.430379867553711]], [[6.806755542755127]], [[6.66092586517334]], [[6.8292365074157715]], [[7.115354061126709]], [[7.544079303741455]], [[6.50487756729126]], [[6.814257621765137]], [[6.6645402908325195]], [[6.562115669250488]], [[7.111221790313721]], [[6.755922794342041]], [[7.447986602783203]], [[6.774875164031982]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_761e0693c7faa426ce89a24460e11a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6251bd591f791eb9838fbe91c37b72d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.566150188446045]], [[2.799795389175415]], [[2.5570602416992188]], [[2.6287052631378174]], [[2.6062774658203125]], [[2.6669459342956543]], [[2.548785924911499]], [[3.1859357357025146]], [[2.9422526359558105]], [[2.615391254425049]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_0b9a3eea6ccb103f24f1590a51c0480e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.051373481750488]], [[4.929189682006836]], [[4.523844242095947]], [[5.058758735656738]], [[4.941986560821533]], [[4.987974166870117]], [[4.642576694488525]], [[4.768600940704346]], [[4.74896240234375]], [[4.5014119148254395]], [[5.0258355140686035]], [[5.735795021057129]], [[5.5453782081604]], [[4.34968376159668]], [[4.784318923950195]], [[5.109044075012207]], [[5.173520088195801]], [[5.17549467086792]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_3931d9aeee5a1aa97b60b925ee8bce59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_831862bc6d25ea81d4a94454e1a13a3c
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.930376052856445, 7.958884239196777, 8.756532669067383, 8.268275260925293, 8.45052719116211, 7.764630317687988, 7.969349384307861, 8.35740852355957, 9.029668807983398, 9.289926528930664, 8.03172492980957, 8.313129425048828, 8.658809661865234, 8.234850883483887, 9.34371280670166, 8.602219581604004, 8.031670570373535, 8.66115951538086, 8.051887512207031, 9.021086692810059, 7.82981014251709, 8.385726928710938, 7.536729335784912, 8.938192367553711, 7.878767013549805, 9.028979301452637, 9.251540184020996, 8.633338928222656, 8.135558128356934, 7.85642671585083]], dtype='float32').reshape([1, 30]),
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


    class TestPrimitiveOp_87cfe752724a617be60fb252531ddc7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.967134952545166]], [[7.665541172027588]], [[7.688358783721924]], [[8.009965896606445]], [[8.143436431884766]], [[8.406558990478516]], [[9.000687599182129]], [[7.82932710647583]], [[7.898717403411865]], [[7.775198936462402]], [[7.320363998413086]], [[6.782022476196289]], [[7.875368118286133]], [[8.605768203735352]], [[8.65597152709961]], [[7.1342549324035645]], [[7.60582160949707]], [[7.848722457885742]], [[8.213285446166992]], [[8.337427139282227]], [[7.865445137023926]], [[7.648224830627441]], [[8.587637901306152]], [[7.777416706085205]], [[8.325520515441895]], [[8.016615867614746]], [[8.52187442779541]], [[7.837628364562988]], [[8.583416938781738]], [[7.866849899291992]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_2c8f524b7ef8504d036271081f398e04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa6a208551763b029a4175fcd015eae
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8248137831687927]], [[1.2744824886322021]], [[1.3420348167419434]], [[1.3940320014953613]], [[1.0370404720306396]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_b1bbfd062cac52bb6cb0c2f55536d195(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.8600144386291504]], [[2.6740565299987793]], [[2.925100088119507]], [[2.906919002532959]], [[2.801386833190918]], [[2.7032203674316406]], [[3.2327170372009277]], [[1.9958714246749878]], [[2.505014181137085]], [[2.3930094242095947]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_511379b66a8b49ab441044e968cd623f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.5436859130859375]], [[5.52134370803833]], [[5.874619483947754]], [[5.935561180114746]], [[6.13942813873291]], [[5.741078853607178]], [[6.110844135284424]], [[5.217152118682861]], [[5.579229354858398]], [[5.655007362365723]], [[6.007051944732666]], [[5.698542594909668]], [[5.283339500427246]], [[6.308893203735352]], [[6.3184733390808105]], [[5.412240505218506]], [[5.156081199645996]], [[6.472560882568359]], [[5.587553977966309]], [[5.511021614074707]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf554d7034bfddca8caf9a93177a7146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.9886651039123535]], [[3.6906166076660156]], [[3.822950601577759]], [[3.7890625]], [[4.127457618713379]], [[3.757260322570801]], [[4.472038745880127]], [[3.595747947692871]], [[4.517457008361816]], [[3.7042181491851807]], [[3.96732234954834]], [[4.572327613830566]], [[4.259017467498779]], [[3.8095781803131104]], [[4.457819938659668]], [[3.9950358867645264]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_fd21ef6117333d607742f59b8972a3f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa5522cc0fff55438ea3c29c97b2341b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3726940155029297]], [[3.6283209323883057]], [[3.0741968154907227]], [[3.126549243927002]], [[2.7311394214630127]], [[3.133653402328491]], [[3.4546656608581543]], [[2.7111501693725586]], [[3.43560791015625]], [[3.4273197650909424]], [[3.554928779602051]], [[2.9367902278900146]], [[3.5619406700134277]], [[3.9031782150268555]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_fdff1b9bc419613550459cc425269bc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.941735744476318]], [[5.350258827209473]], [[4.993419170379639]], [[4.997262954711914]], [[4.465219020843506]], [[5.24635648727417]], [[4.697439193725586]], [[4.72060489654541]], [[4.484288215637207]], [[4.6957550048828125]], [[4.52725076675415]], [[5.010762691497803]], [[4.304614543914795]], [[4.8619818687438965]], [[4.338233470916748]], [[4.847714900970459]], [[4.7345099449157715]], [[5.4691667556762695]], [[5.441762924194336]], [[4.312991619110107]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_12a5227a9b0b79f7159e55cd3464d091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.716361999511719]], [[8.297204971313477]], [[7.497693061828613]], [[7.400525093078613]], [[8.344557762145996]], [[8.721635818481445]], [[7.63230037689209]], [[7.712268829345703]], [[8.096673965454102]], [[7.462275505065918]], [[8.959723472595215]], [[7.646790027618408]], [[7.697815418243408]], [[8.115347862243652]], [[7.619071960449219]], [[7.572359085083008]], [[7.913558006286621]], [[7.5322418212890625]], [[7.958657741546631]], [[7.420487880706787]], [[8.024356842041016]], [[8.10165023803711]], [[7.993789196014404]], [[7.642430782318115]], [[7.333017826080322]], [[7.333766937255859]], [[8.388383865356445]], [[7.282675743103027]], [[7.608148574829102]], [[8.325189590454102]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_9cae39406d4a8cfc613a06e66fa40c14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.591955184936523]], [[5.937971591949463]], [[5.838271141052246]], [[5.930294990539551]], [[5.167905807495117]], [[5.86020565032959]], [[5.7827863693237305]], [[6.046093940734863]], [[5.75948429107666]], [[5.110523223876953]], [[6.796349048614502]], [[6.009696006774902]], [[6.375565528869629]], [[6.133172512054443]], [[5.7320027351379395]], [[5.841438293457031]], [[5.657365322113037]], [[5.535276889801025]], [[5.8343400955200195]], [[5.474064826965332]], [[5.478018760681152]], [[5.710632801055908]], [[6.113531589508057]], [[6.6900506019592285]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f9b7d234b4c7b369bc6dbf8f8f6e626c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.309843063354492]], [[6.539846897125244]], [[6.768796920776367]], [[6.078126430511475]], [[7.450684070587158]], [[5.8743414878845215]], [[6.8607683181762695]], [[6.394069194793701]], [[6.257390975952148]], [[6.216409683227539]], [[6.404240608215332]], [[6.725374221801758]], [[6.286440849304199]], [[6.559449195861816]], [[7.190722942352295]], [[7.1903815269470215]], [[7.360578536987305]], [[6.062586307525635]], [[6.55659294128418]], [[6.15986967086792]], [[6.108526229858398]], [[5.927225112915039]], [[6.7410383224487305]], [[6.556881427764893]], [[7.01950216293335]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_330c6332ac259be83473fcb3ca33e585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6438512802124023]], [[3.296189785003662]], [[2.9459354877471924]], [[3.5804953575134277]], [[3.079784631729126]], [[2.939220666885376]], [[3.0548999309539795]], [[2.908855676651001]], [[3.5852274894714355]], [[3.0470340251922607]], [[3.243905544281006]], [[3.517317533493042]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


    class TestPrimitiveOp_70b50e3c65b16cff6f789b9ab570cbdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[709.1481323242188]], [[692.7680053710938]], [[735.9580688476562]], [[664.4910278320312]], [[767.9503784179688]], [[734.5795288085938]], [[688.29736328125]], [[765.3384399414062]], [[746.7483520507812]], [[764.594482421875]], [[721.3216552734375]], [[714.5488891601562]], [[668.8396606445312]], [[734.5792846679688]], [[712.2596435546875]], [[680.5489501953125]], [[642.421875]], [[701.2931518554688]], [[649.7561645507812]], [[729.9595336914062]], [[730.0673217773438]], [[656.1763916015625]], [[681.51171875]], [[740.1054077148438]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_3a5d81d011a68a2c5af5680488ea81a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[88.58912658691406]], [[85.60018920898438]], [[74.9617919921875]], [[79.29644775390625]], [[82.23506927490234]], [[86.25605773925781]], [[86.15576171875]], [[92.4328842163086]], [[82.28874206542969]], [[89.9693603515625]], [[81.6507339477539]], [[84.16071319580078]], [[71.01463317871094]], [[85.02629089355469]], [[81.29607391357422]], [[79.44832611083984]], [[79.5385513305664]], [[87.98348236083984]], [[83.3065414428711]], [[81.68899536132812]], [[73.86360931396484]], [[76.48143005371094]], [[71.70701599121094]], [[85.40310668945312]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_d30551f2365c25fe5e327f31d5c717e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45.137916564941406]], [[44.4119987487793]], [[37.813167572021484]], [[35.5390625]], [[45.238399505615234]], [[44.228973388671875]], [[43.900428771972656]], [[49.975494384765625]], [[42.41387176513672]], [[41.04962921142578]], [[38.83075714111328]], [[42.41850662231445]], [[39.964046478271484]], [[47.64216232299805]], [[47.3900260925293]], [[46.39550018310547]], [[43.45512771606445]], [[44.626033782958984]], [[42.45619201660156]], [[41.56513595581055]], [[45.767799377441406]], [[46.025306701660156]], [[41.83007049560547]], [[42.63505935668945]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_00e8eb481d3b1d1e9f43a67d605af200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[23.270776748657227]], [[19.274690628051758]], [[19.025012969970703]], [[20.659255981445312]], [[21.67249870300293]], [[22.679594039916992]], [[22.12632179260254]], [[19.17441749572754]], [[20.16863250732422]], [[21.49508285522461]], [[22.29465103149414]], [[22.043739318847656]], [[21.93906593322754]], [[20.350486755371094]], [[21.37721824645996]], [[22.92276382446289]], [[21.07513427734375]], [[23.164953231811523]], [[20.970190048217773]], [[23.452327728271484]], [[20.533618927001953]], [[21.655485153198242]], [[19.653417587280273]], [[21.133453369140625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_976f57e5db8b3e075c4e88507121858c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34209.76953125]], [[35293.09375]], [[36197.36328125]], [[41263.84765625]], [[33075.9453125]], [[39370.27734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_896ba8d1b0f89a41d683a4e17c6eab77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43286.53515625]], [[49013.6796875]], [[40533.49609375]], [[43197.0859375]], [[38508.86328125]], [[35771.30078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_6d34f1c384648ccb5b2dc7af48228609(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37526.0234375]], [[36022.96875]], [[39461.9140625]], [[41080.35546875]], [[40742.5625]], [[35906.984375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_18a536e951f9db3f78e7a58753732995(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[40397.62109375]], [[37758.46875]], [[48596.2578125]], [[39773.49609375]], [[43267.83203125]], [[38806.35546875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


    class TestPrimitiveOp_8f4855c9cb1e56d4290de21d5ba0aedf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.869830131530762]], [[6.091160297393799]], [[6.01346492767334]], [[5.536749362945557]], [[6.523960113525391]], [[6.541598320007324]], [[6.203537464141846]], [[5.59697961807251]], [[5.732237815856934]], [[6.437671184539795]], [[5.600796222686768]], [[5.776000499725342]], [[6.717878818511963]], [[5.755391597747803]], [[6.647819519042969]], [[5.8235297203063965]], [[5.793607234954834]], [[6.186882495880127]], [[6.169209957122803]], [[5.716612815856934]], [[6.256058216094971]], [[5.74980354309082]], [[6.225977897644043]], [[5.876616477966309]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_87fc85255706319020a720a6a2f079f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa5482916cffc526445466b37baf168
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.33536958694458, 5.728888511657715, 4.716067790985107, 4.229830265045166, 5.404992580413818, 4.498203754425049, 5.148319721221924, 4.750048637390137, 5.1595282554626465, 4.218147277832031, 5.205923080444336, 4.9782233238220215, 4.736413478851318, 4.962367057800293, 5.054198265075684, 5.145442008972168, 4.78067684173584, 4.842703819274902]], dtype='float32').reshape([1, 18]),
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


    class TestPrimitiveOp_8422853cfb3d9edb4a7d2250696ccc00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e722c534a85f6265bab20f6209cc641
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.162670612335205, 4.743340492248535, 5.998516082763672, 5.040639877319336, 5.3741350173950195, 5.7692341804504395, 5.374410152435303, 5.690393447875977, 6.033936023712158, 5.206454277038574, 5.5260396003723145, 5.6504645347595215, 5.447342395782471, 5.529510974884033, 5.592252254486084, 6.441046714782715, 5.761960983276367, 5.791680812835693, 5.865000247955322, 5.504603385925293, 5.820616245269775, 5.220635890960693, 5.280458450317383]], dtype='float32').reshape([1, 23]),
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


    class TestPrimitiveOp_a4d5d06ab90971590073a2612616919b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.156549453735352]], [[6.578279972076416]], [[8.965864181518555]], [[7.614697456359863]], [[7.707426071166992]], [[7.436953067779541]], [[7.509988307952881]], [[7.788017749786377]], [[6.393144607543945]], [[7.362917900085449]], [[7.2338972091674805]], [[8.074005126953125]], [[7.098735809326172]], [[6.910137176513672]], [[8.077880859375]], [[8.106316566467285]], [[7.840653419494629]], [[8.020371437072754]], [[8.272087097167969]], [[8.352813720703125]], [[7.642253875732422]], [[7.687630653381348]], [[7.613253593444824]], [[7.520717620849609]], [[7.637831687927246]], [[8.50828742980957]], [[6.867826461791992]], [[7.150737762451172]], [[7.4321746826171875]], [[7.626999378204346]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_5f9918ad27705f445836bd650bb71762(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.764479637145996]], [[8.717795372009277]], [[8.243857383728027]], [[6.911282062530518]], [[7.926513671875]], [[8.030611991882324]], [[8.945138931274414]], [[7.763011932373047]], [[7.695864200592041]], [[7.878115177154541]], [[8.235343933105469]], [[6.995920658111572]], [[8.233352661132812]], [[7.727132797241211]], [[7.220941066741943]], [[8.071619033813477]], [[8.144110679626465]], [[6.693134307861328]], [[7.199584484100342]], [[8.235687255859375]], [[7.900215148925781]], [[7.703789234161377]], [[7.636146545410156]], [[8.11130428314209]], [[8.535118103027344]], [[7.386641502380371]], [[8.645404815673828]], [[6.932118892669678]], [[7.863061904907227]], [[8.122811317443848]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_86f35e645d08298bcb4b6bc0caaacd5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8566051721572876]], [[2.095454216003418]], [[1.8238812685012817]], [[1.474919080734253]], [[1.5276159048080444]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_8281dd9eeb7f20e285b1581e9868ec6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.719106674194336]], [[2.747519016265869]], [[2.2771918773651123]], [[2.710972547531128]], [[2.6780829429626465]], [[2.6255321502685547]], [[2.8236663341522217]], [[2.8805809020996094]], [[2.647463321685791]], [[2.6628236770629883]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_2a87bc090b00cbe8b87ac4ac57f0a7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.086631774902344]], [[7.308384895324707]], [[5.830771446228027]], [[6.009194850921631]], [[5.87945032119751]], [[6.114314079284668]], [[5.7845635414123535]], [[6.975741386413574]], [[5.626193523406982]], [[6.476055145263672]], [[6.509440898895264]], [[6.814225673675537]], [[6.179939270019531]], [[6.212969779968262]], [[5.426244735717773]], [[5.825172424316406]], [[6.444512367248535]], [[6.775128364562988]], [[5.420651435852051]], [[6.370400428771973]], [[6.124203205108643]], [[6.66068172454834]], [[6.493947505950928]], [[6.2271623611450195]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_a0daa9fcf2d9ebe6c280d5ceaf764d5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.429371356964111]], [[5.102461814880371]], [[5.122546195983887]], [[4.959865093231201]], [[4.487847328186035]], [[4.595395565032959]], [[4.401482582092285]], [[5.045962333679199]], [[5.277590274810791]], [[4.407054424285889]], [[4.078505992889404]], [[4.891146183013916]], [[4.863668918609619]], [[4.882938385009766]], [[4.876100540161133]], [[4.58686637878418]], [[4.975370407104492]], [[5.126645565032959]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57873602ae5ed23e54b2353850c7d415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.2779717445373535]], [[6.342955589294434]], [[6.889688014984131]], [[6.494229316711426]], [[6.744277000427246]], [[6.069737911224365]], [[7.056484222412109]], [[5.741796493530273]], [[6.317704677581787]], [[6.743081569671631]], [[6.25610876083374]], [[6.39182186126709]], [[5.538364410400391]], [[6.538943290710449]], [[6.79679536819458]], [[7.517889022827148]], [[6.390634536743164]], [[6.382289886474609]], [[6.212860584259033]], [[6.586852073669434]], [[5.402479648590088]], [[6.590733051300049]], [[6.075348854064941]], [[5.696362495422363]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_41cbd4fc6e5f6b0d9c5c6fb286de5008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9a5be699233ae66d2644ef9ba39603
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5981800556182861]], [[1.4517943859100342]], [[0.8325352668762207]], [[1.1580177545547485]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_4e5fca1be4380d7fff1bd635066cdcdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85b01dc5b87514b414a7a8aee3b35d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.759239673614502]], [[3.067654848098755]], [[2.8620004653930664]], [[2.730835437774658]], [[2.2491555213928223]], [[2.969370126724243]], [[3.7615504264831543]], [[2.9485251903533936]], [[2.366492748260498]], [[3.176340103149414]], [[2.972701072692871]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_d4263d93b8c570314266f7d029d2e6d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[9.219853401184082]], [[7.977720737457275]], [[7.273802280426025]], [[8.22716236114502]], [[8.4916410446167]], [[9.26848316192627]], [[7.195425987243652]], [[7.818164825439453]], [[8.419690132141113]], [[7.483343601226807]], [[8.165380477905273]], [[7.699680328369141]], [[8.12510871887207]], [[8.931337356567383]], [[7.405519485473633]], [[9.185028076171875]], [[7.8361005783081055]], [[7.829963684082031]], [[8.57872200012207]], [[7.620147705078125]], [[8.942919731140137]], [[7.511859893798828]], [[7.994983196258545]], [[7.808498859405518]], [[8.88344955444336]], [[7.918191432952881]], [[8.144401550292969]], [[8.468025207519531]], [[8.549529075622559]], [[8.860572814941406]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_6b0f6bc232138fd96fc7e12996aa6b22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.339223384857178]], [[4.631780624389648]], [[4.155935764312744]], [[3.9742817878723145]], [[5.302306652069092]], [[4.456806659698486]], [[4.654520034790039]], [[4.203819751739502]], [[4.233545303344727]], [[3.887152910232544]], [[4.019169807434082]], [[4.785059452056885]], [[5.198037147521973]], [[4.3127031326293945]], [[3.808553695678711]], [[4.056495666503906]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_fa10687558e69e65dcae3c7a7cd81fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.9523186683654785]], [[7.42242431640625]], [[8.545364379882812]], [[8.203980445861816]], [[7.4313483238220215]], [[7.6026763916015625]], [[8.048667907714844]], [[8.16807746887207]], [[7.477578163146973]], [[7.682559967041016]], [[7.5439887046813965]], [[8.519378662109375]], [[8.315771102905273]], [[7.916327476501465]], [[7.257624626159668]], [[8.291592597961426]], [[7.157421588897705]], [[7.265719890594482]], [[7.141048908233643]], [[7.891236305236816]], [[8.022711753845215]], [[8.24140453338623]], [[7.856504917144775]], [[8.895251274108887]], [[8.04963493347168]], [[8.79507064819336]], [[8.328497886657715]], [[7.987967491149902]], [[8.476158142089844]], [[7.17471170425415]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_c35164081ac84fc02cc52b80c1b162a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.717169761657715]], [[6.377462387084961]], [[6.11559534072876]], [[6.248219966888428]], [[6.040029048919678]], [[7.45786714553833]], [[7.106032371520996]], [[5.864388465881348]], [[6.851099014282227]], [[6.623362064361572]], [[6.710056781768799]], [[5.612614154815674]], [[7.330811023712158]], [[5.56170129776001]], [[7.341120719909668]], [[6.5250678062438965]], [[6.4778265953063965]], [[6.345496654510498]], [[7.3401031494140625]], [[6.775630474090576]], [[6.1982197761535645]], [[5.557824611663818]], [[6.792047023773193]], [[6.723367214202881]], [[6.0354838371276855]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_d726c02ffe0fe4c7f61c1a60a482d56e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.431391716003418]], [[4.728453636169434]], [[3.8485469818115234]], [[3.7370498180389404]], [[4.240548133850098]], [[3.7491250038146973]], [[4.5072808265686035]], [[4.023393154144287]], [[3.9357423782348633]], [[4.149059295654297]], [[4.889106273651123]], [[4.266361236572266]], [[4.887792587280273]], [[4.657820701599121]], [[4.33989143371582]], [[4.628081321716309]], [[3.937661647796631]], [[4.499518394470215]], [[4.870829105377197]], [[4.252282619476318]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_c15e3ddaa4d56b7bbc5ef254bb111028(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.285511016845703]], [[4.616420745849609]], [[5.1114630699157715]], [[4.157078742980957]], [[4.800133228302002]], [[3.777223587036133]], [[4.694716930389404]], [[3.983590841293335]], [[3.9912760257720947]], [[4.235405921936035]], [[4.37108039855957]], [[4.672853469848633]], [[4.280800819396973]], [[4.807054042816162]], [[4.614780426025391]], [[4.6422810554504395]], [[4.211650371551514]], [[5.195508003234863]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_68758d5cab1700d6aaef46a17bf65441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.78338623046875]], [[4.758307456970215]], [[4.5570759773254395]], [[4.651852607727051]], [[4.396996021270752]], [[4.2491278648376465]], [[4.278838634490967]], [[4.917035102844238]], [[4.2762885093688965]], [[4.933725833892822]], [[5.054064750671387]], [[4.185392379760742]], [[4.766103267669678]], [[4.256556510925293]], [[4.734835147857666]], [[5.001825332641602]], [[3.9590556621551514]], [[4.636456489562988]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_177aab49fa33a6f90da440e5529d03ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc8405c8dc00d11843d8deda02d87197
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1a88fcb1e752d409135b18ec0cd3e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.475754737854004]], [[5.490368366241455]], [[5.066490650177002]], [[6.535336494445801]], [[6.574573993682861]], [[6.074638843536377]], [[5.812880039215088]], [[6.25173282623291]], [[6.593845367431641]], [[6.095102787017822]], [[6.271059036254883]], [[5.55356502532959]], [[6.959198951721191]], [[5.8360443115234375]], [[6.394191741943359]], [[5.082563400268555]], [[5.476058006286621]], [[5.893261909484863]], [[5.3520402908325195]], [[6.88227653503418]], [[6.345145225524902]], [[5.516343116760254]], [[6.1561150550842285]], [[6.437520980834961]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_d40b28de3186421dac957c199ef56536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.772536277770996]], [[5.173009395599365]], [[3.8997905254364014]], [[5.0453996658325195]], [[4.3154191970825195]], [[5.118714809417725]], [[4.704500198364258]], [[4.647329807281494]], [[4.215498924255371]], [[4.829869270324707]], [[4.632633686065674]], [[5.515543460845947]], [[4.459994792938232]], [[4.442266464233398]], [[4.744085788726807]], [[4.792042255401611]], [[4.764348030090332]], [[4.661513805389404]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_d8e575382975dbe7b81ff2d28d7d683f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.520150184631348]], [[4.817781925201416]], [[4.683832168579102]], [[5.0453901290893555]], [[4.88814115524292]], [[3.796255111694336]], [[5.0784525871276855]], [[4.307088375091553]], [[4.392723560333252]], [[5.034638404846191]], [[4.890681266784668]], [[4.789068222045898]], [[4.620427131652832]], [[4.711824893951416]], [[4.296992778778076]], [[4.6062116622924805]], [[4.63204288482666]], [[5.542410373687744]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_64626ed70e26d0a1394234ea511800e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.531711101531982]], [[4.344454288482666]], [[4.5150837898254395]], [[4.630534648895264]], [[4.285594463348389]], [[4.032207489013672]], [[3.536329746246338]], [[4.6597137451171875]], [[4.211676597595215]], [[4.191618919372559]], [[4.450105667114258]], [[5.07877779006958]], [[4.409242153167725]], [[4.230588436126709]], [[4.868673324584961]], [[4.5204758644104]], [[5.747686386108398]], [[4.947834491729736]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_bd8f088c8376a1ea4366151de05dbe28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.25018835067749]], [[4.311614513397217]], [[4.094573497772217]], [[3.818601131439209]], [[3.6546530723571777]], [[3.475036144256592]], [[3.1861672401428223]], [[3.8367486000061035]], [[4.038932800292969]], [[4.63492488861084]], [[3.6220250129699707]], [[3.9613070487976074]], [[3.4617204666137695]], [[4.631092548370361]], [[3.767266273498535]], [[4.41674280166626]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_c3d75c61ddff32bdc6395f7ba49986d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.675265789031982]], [[4.467334270477295]], [[3.612835645675659]], [[4.771159648895264]], [[4.658884525299072]], [[4.322079658508301]], [[4.130802631378174]], [[4.945823669433594]], [[4.1770548820495605]], [[4.741894245147705]], [[4.489341735839844]], [[4.604240417480469]], [[4.373265266418457]], [[3.9902279376983643]], [[4.216479778289795]], [[4.061892986297607]], [[4.76384973526001]], [[3.904416561126709]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_d1cda700c8d505bde1eb5f6f98284357(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9a5be699233ae66d2644ef9ba39603
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9944493174552917]], [[1.2166593074798584]], [[1.1984256505966187]], [[1.1523396968841553]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_6a640e23a1b3264aec992815f18c4d8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.8062310218811035]], [[5.361790180206299]], [[4.5029802322387695]], [[4.500344753265381]], [[4.982926845550537]], [[4.848570346832275]], [[4.778101921081543]], [[5.076159954071045]], [[4.993417263031006]], [[4.963909149169922]], [[4.629103660583496]], [[4.729391098022461]], [[5.710188865661621]], [[4.812963008880615]], [[5.511858940124512]], [[5.036686420440674]], [[5.425128936767578]], [[4.4527363777160645]], [[5.328535556793213]], [[5.170138359069824]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_22bd09fbbfee5148a6687182a90413e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.342505693435669]], [[2.7520391941070557]], [[2.714287519454956]], [[3.211639881134033]], [[3.0043015480041504]], [[3.492180347442627]], [[3.593163013458252]], [[3.679666042327881]], [[2.5081582069396973]], [[3.2115612030029297]], [[3.288628101348877]], [[3.0804831981658936]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_52f75d5dffaf74ab1e0c0ea0be417015(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.7912914752960205]], [[4.724730014801025]], [[5.298625946044922]], [[5.1255106925964355]], [[4.62327241897583]], [[5.113589286804199]], [[5.026547431945801]], [[4.239257335662842]], [[4.4622602462768555]], [[5.033857822418213]], [[3.9976367950439453]], [[4.5428996086120605]], [[4.076603889465332]], [[4.696175575256348]], [[4.704270839691162]], [[4.375837326049805]], [[4.985689640045166]], [[4.377310276031494]], [[4.806520938873291]], [[4.063906669616699]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_095a3909dce9a7e4b66f42b49503e82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85b01dc5b87514b414a7a8aee3b35d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0060856342315674]], [[2.823049545288086]], [[2.855907440185547]], [[3.3607177734375]], [[3.1934914588928223]], [[3.269962787628174]], [[2.899298667907715]], [[3.100433111190796]], [[3.093851089477539]], [[2.913404703140259]], [[2.969477653503418]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_585a5191f1ba443d69c554368f96d6c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96dc8643dc29e249b7d4dda0732345c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.2398808002471924]], [[3.508150577545166]], [[2.8001081943511963]], [[3.5751781463623047]], [[2.6233019828796387]], [[3.1378047466278076]], [[3.546370506286621]], [[3.4879634380340576]], [[3.134951114654541]], [[3.1070199012756348]], [[3.4020214080810547]], [[2.941101312637329]], [[3.3202311992645264]], [[3.3293280601501465]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


    class TestPrimitiveOp_09feefdb39753fe87363fbfd9a11cc94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.995167255401611]], [[5.068471908569336]], [[5.232618808746338]], [[4.3133463859558105]], [[5.070679664611816]], [[4.943248271942139]], [[5.22681188583374]], [[4.9932475090026855]], [[5.181914806365967]], [[4.9883599281311035]], [[4.587845325469971]], [[4.951524257659912]], [[4.5863518714904785]], [[4.5127482414245605]], [[4.808051109313965]], [[5.10516881942749]], [[4.906620979309082]], [[4.632011413574219]], [[4.7547783851623535]], [[4.731569766998291]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_a7c2a52890fc84ae54eb84973598b37a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43951.48828125]], [[34423.453125]], [[34047.71484375]], [[29783.375]], [[37019.40234375]], [[38497.50390625]]], [[[43313.31640625]], [[33921.72265625]], [[33559.36328125]], [[29355.10546875]], [[36486.9140625]], [[37942.50390625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_83dde7d8dd90cc86788546476e68fdf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37845.80859375]], [[32601.22265625]], [[41104.3828125]], [[33579.23828125]], [[41183.9921875]], [[40182.13671875]]], [[[36832.75390625]], [[31724.77734375]], [[40006.65625]], [[32682.271484375]], [[40081.921875]], [[39108.6875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_e1f19a31085a0fa37cf5735afdbd1b9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[47123.75]], [[41793.8828125]], [[49294.4765625]], [[44015.36328125]], [[40978.10546875]], [[35567.5703125]]], [[[45730.7265625]], [[40560.58203125]], [[47835.48828125]], [[42716.0703125]], [[39762.51953125]], [[34511.59375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_7a62809811fe5b845204da6ad1bce0d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37736.85546875]], [[39921.3515625]], [[44999.49609375]], [[36545.47265625]], [[40640.953125]], [[44243.4921875]]], [[[36904.62109375]], [[39050.265625]], [[44017.51171875]], [[35748.12890625]], [[39746.23828125]], [[43278.5859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


    class TestPrimitiveOp_7cc63cdbef99c5054b164f30059f1871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.301014423370361]], [[8.403518676757812]], [[8.03527545928955]], [[8.760984420776367]], [[8.188155174255371]], [[7.942190170288086]], [[7.466071128845215]], [[7.920556545257568]], [[8.333847045898438]], [[8.276917457580566]], [[6.924700736999512]], [[7.403339385986328]], [[7.732781410217285]], [[8.625016212463379]], [[6.764622688293457]], [[7.116641044616699]], [[8.37065601348877]], [[8.823826789855957]], [[8.26421070098877]], [[8.392292022705078]], [[6.971035480499268]], [[7.877248287200928]], [[8.675994873046875]], [[7.90565299987793]], [[8.166635513305664]], [[8.00434684753418]], [[7.3291497230529785]], [[7.97578239440918]], [[7.904312610626221]], [[8.364042282104492]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_7285bdcb5bc1187974058df26a5108ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.341287136077881]], [[7.299825191497803]], [[8.345277786254883]], [[7.708090782165527]], [[6.989670753479004]], [[7.035129070281982]], [[7.876548767089844]], [[7.641441822052002]], [[7.257607460021973]], [[7.430710792541504]], [[7.43454647064209]], [[7.339380741119385]], [[7.949814319610596]], [[7.445698261260986]], [[7.395953178405762]], [[7.797006130218506]], [[7.37500524520874]], [[7.733424186706543]], [[7.49075984954834]], [[7.240258693695068]], [[7.815331935882568]], [[8.243738174438477]], [[7.368047714233398]], [[7.8810553550720215]], [[8.4630126953125]], [[7.198576927185059]], [[6.862584114074707]], [[8.420339584350586]], [[7.592401504516602]], [[7.650505542755127]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_21d6aa38c408ccbd4c6fa63aa5017972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.334691047668457]], [[6.553110122680664]], [[7.550365924835205]], [[7.175827980041504]], [[7.56691837310791]], [[7.009180545806885]], [[7.625830173492432]], [[6.939151763916016]], [[7.596673488616943]], [[6.650327682495117]], [[7.773587703704834]], [[7.105720043182373]], [[7.4982757568359375]], [[6.71769905090332]], [[7.4405083656311035]], [[7.593581199645996]], [[7.8551530838012695]], [[7.435949802398682]], [[8.307052612304688]], [[7.227053642272949]], [[7.727344036102295]], [[7.8308796882629395]], [[7.484035968780518]], [[6.645956516265869]], [[7.704050064086914]], [[7.600241184234619]], [[6.777804851531982]], [[7.476434230804443]], [[7.012086868286133]], [[7.311110496520996]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_a6568af647de442ff43c9881bb6e664a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.924079895019531]], [[7.3153605461120605]], [[7.995610237121582]], [[7.2530198097229]], [[7.813193321228027]], [[7.922425270080566]], [[7.419620990753174]], [[7.603722095489502]], [[7.701774597167969]], [[7.354045867919922]], [[7.252942085266113]], [[8.358896255493164]], [[7.384546756744385]], [[8.001121520996094]], [[7.920747756958008]], [[7.5409440994262695]], [[7.896615505218506]], [[8.030363082885742]], [[8.464024543762207]], [[8.035039901733398]], [[7.323920726776123]], [[7.297919750213623]], [[7.611123085021973]], [[7.6905646324157715]], [[6.916043281555176]], [[6.890603542327881]], [[7.488044261932373]], [[6.946822166442871]], [[7.933925151824951]], [[7.751521587371826]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_53e58f00ffc83ae53db7870df08380ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.44456148147583]], [[3.229776382446289]], [[3.124418258666992]], [[3.338461399078369]], [[3.577549457550049]], [[3.919827699661255]], [[3.9549551010131836]], [[3.856282949447632]], [[4.125796318054199]], [[3.8217051029205322]], [[2.8720881938934326]], [[3.986161947250366]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_605a999f84714168749555ea37c3e64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.021803140640259]], [[2.8979740142822266]], [[2.6711370944976807]], [[3.363351345062256]], [[2.642312526702881]], [[3.2148635387420654]], [[3.0581884384155273]], [[3.5348715782165527]], [[3.4332950115203857]], [[3.799393653869629]], [[3.1188201904296875]], [[2.997915029525757]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_baf10e83e7e528c3c3a7bcd0780a5ce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.129674911499023]], [[6.374298095703125]], [[6.090598106384277]], [[6.553129196166992]], [[6.5286946296691895]], [[6.757339000701904]], [[6.135092735290527]], [[6.8264689445495605]], [[6.315954685211182]], [[7.21508264541626]], [[6.141348838806152]], [[5.7715253829956055]], [[6.627416610717773]], [[6.3487701416015625]], [[6.562192916870117]], [[6.3334736824035645]], [[6.540079116821289]], [[6.168745517730713]], [[6.445700645446777]], [[6.114558696746826]], [[5.674906253814697]], [[6.7818779945373535]], [[6.279696941375732]], [[6.37258243560791]], [[6.395266532897949]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_4820d5097d6c1c491dcb87da1f0bc284(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.234936237335205]], [[4.312356948852539]], [[4.602973937988281]], [[4.165308952331543]], [[4.123189926147461]], [[4.576277732849121]], [[4.517114639282227]], [[4.990133285522461]], [[4.928924083709717]], [[4.565418720245361]], [[4.6072797775268555]], [[4.104726791381836]], [[4.53456974029541]], [[4.675724983215332]], [[5.090818405151367]], [[4.430437088012695]], [[4.960774898529053]], [[4.91608190536499]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_aed6197d24914caf579adb50cfacd127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.412215232849121]], [[1.8416154384613037]], [[1.87169349193573]], [[1.7094300985336304]], [[1.4866801500320435]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_14351e0caccc92af5479539218975ddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6466522216796875]], [[3.1592297554016113]], [[2.6499240398406982]], [[3.1042873859405518]], [[2.4753024578094482]], [[2.788867950439453]], [[3.2296197414398193]], [[2.8998422622680664]], [[2.356828212738037]], [[2.5325305461883545]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_98e0f932698350d637eec48b8bab85c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.219168663024902]], [[4.386056900024414]], [[4.218252658843994]], [[3.8906869888305664]], [[3.821678876876831]], [[5.3758955001831055]], [[4.510594844818115]], [[4.741876125335693]], [[4.260367393493652]], [[3.830246686935425]], [[4.32592248916626]], [[4.207342147827148]], [[4.228832244873047]], [[4.594235420227051]], [[4.741360664367676]], [[4.85493803024292]], [[4.056390762329102]], [[4.101260185241699]], [[5.8075480461120605]], [[4.5099711418151855]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_5d73ac3638a483dc43e6a78069dbb470(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.851374626159668]], [[6.279842853546143]], [[6.162980556488037]], [[6.155672550201416]], [[6.570970058441162]], [[6.711134910583496]], [[6.90558385848999]], [[6.065819263458252]], [[7.283596992492676]], [[7.3350701332092285]], [[7.430379867553711]], [[6.806755542755127]], [[6.66092586517334]], [[6.8292365074157715]], [[7.115354061126709]], [[7.544079303741455]], [[6.50487756729126]], [[6.814257621765137]], [[6.6645402908325195]], [[6.562115669250488]], [[7.111221790313721]], [[6.755922794342041]], [[7.447986602783203]], [[6.774875164031982]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_71c95d90554de7ec87f53b988110809b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.566150188446045]], [[2.799795389175415]], [[2.5570602416992188]], [[2.6287052631378174]], [[2.6062774658203125]], [[2.6669459342956543]], [[2.548785924911499]], [[3.1859357357025146]], [[2.9422526359558105]], [[2.615391254425049]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_0a42d6997bbed9880668228be4c999a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.051373481750488]], [[4.929189682006836]], [[4.523844242095947]], [[5.058758735656738]], [[4.941986560821533]], [[4.987974166870117]], [[4.642576694488525]], [[4.768600940704346]], [[4.74896240234375]], [[4.5014119148254395]], [[5.0258355140686035]], [[5.735795021057129]], [[5.5453782081604]], [[4.34968376159668]], [[4.784318923950195]], [[5.109044075012207]], [[5.173520088195801]], [[5.17549467086792]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_19f0406e802afe21a0aa46f60d2c4b3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c2d4786102bcc3bb26974ba10e39c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.930376052856445, 7.958884239196777, 8.756532669067383, 8.268275260925293, 8.45052719116211, 7.764630317687988, 7.969349384307861, 8.35740852355957, 9.029668807983398, 9.289926528930664, 8.03172492980957, 8.313129425048828, 8.658809661865234, 8.234850883483887, 9.34371280670166, 8.602219581604004, 8.031670570373535, 8.66115951538086, 8.051887512207031, 9.021086692810059, 7.82981014251709, 8.385726928710938, 7.536729335784912, 8.938192367553711, 7.878767013549805, 9.028979301452637, 9.251540184020996, 8.633338928222656, 8.135558128356934, 7.85642671585083]], dtype='float32').reshape([1, 30]),
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


    class TestPrimitiveOp_592c9b997aca8656056bdef94b76d7c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.967134952545166]], [[7.665541172027588]], [[7.688358783721924]], [[8.009965896606445]], [[8.143436431884766]], [[8.406558990478516]], [[9.000687599182129]], [[7.82932710647583]], [[7.898717403411865]], [[7.775198936462402]], [[7.320363998413086]], [[6.782022476196289]], [[7.875368118286133]], [[8.605768203735352]], [[8.65597152709961]], [[7.1342549324035645]], [[7.60582160949707]], [[7.848722457885742]], [[8.213285446166992]], [[8.337427139282227]], [[7.865445137023926]], [[7.648224830627441]], [[8.587637901306152]], [[7.777416706085205]], [[8.325520515441895]], [[8.016615867614746]], [[8.52187442779541]], [[7.837628364562988]], [[8.583416938781738]], [[7.866849899291992]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_ed419d07275d9c0abde8abc48ff57b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8248137831687927]], [[1.2744824886322021]], [[1.3420348167419434]], [[1.3940320014953613]], [[1.0370404720306396]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_c1a58363f34587b57be9ed8aafcd8e90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.8600144386291504]], [[2.6740565299987793]], [[2.925100088119507]], [[2.906919002532959]], [[2.801386833190918]], [[2.7032203674316406]], [[3.2327170372009277]], [[1.9958714246749878]], [[2.505014181137085]], [[2.3930094242095947]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_9d446f8ffd6c7f693620a0988b6c204f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.5436859130859375]], [[5.52134370803833]], [[5.874619483947754]], [[5.935561180114746]], [[6.13942813873291]], [[5.741078853607178]], [[6.110844135284424]], [[5.217152118682861]], [[5.579229354858398]], [[5.655007362365723]], [[6.007051944732666]], [[5.698542594909668]], [[5.283339500427246]], [[6.308893203735352]], [[6.3184733390808105]], [[5.412240505218506]], [[5.156081199645996]], [[6.472560882568359]], [[5.587553977966309]], [[5.511021614074707]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5a387fd9bc925e74d9217176571b45c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.9886651039123535]], [[3.6906166076660156]], [[3.822950601577759]], [[3.7890625]], [[4.127457618713379]], [[3.757260322570801]], [[4.472038745880127]], [[3.595747947692871]], [[4.517457008361816]], [[3.7042181491851807]], [[3.96732234954834]], [[4.572327613830566]], [[4.259017467498779]], [[3.8095781803131104]], [[4.457819938659668]], [[3.9950358867645264]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_a89d11427d474d48413f254ddca2ea77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96dc8643dc29e249b7d4dda0732345c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3726940155029297]], [[3.6283209323883057]], [[3.0741968154907227]], [[3.126549243927002]], [[2.7311394214630127]], [[3.133653402328491]], [[3.4546656608581543]], [[2.7111501693725586]], [[3.43560791015625]], [[3.4273197650909424]], [[3.554928779602051]], [[2.9367902278900146]], [[3.5619406700134277]], [[3.9031782150268555]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_9c3531395f8dd60ea0b7e80c15c8a6ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.941735744476318]], [[5.350258827209473]], [[4.993419170379639]], [[4.997262954711914]], [[4.465219020843506]], [[5.24635648727417]], [[4.697439193725586]], [[4.72060489654541]], [[4.484288215637207]], [[4.6957550048828125]], [[4.52725076675415]], [[5.010762691497803]], [[4.304614543914795]], [[4.8619818687438965]], [[4.338233470916748]], [[4.847714900970459]], [[4.7345099449157715]], [[5.4691667556762695]], [[5.441762924194336]], [[4.312991619110107]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_d2410fd472b4ad8085df8e47feb5b705(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.716361999511719]], [[8.297204971313477]], [[7.497693061828613]], [[7.400525093078613]], [[8.344557762145996]], [[8.721635818481445]], [[7.63230037689209]], [[7.712268829345703]], [[8.096673965454102]], [[7.462275505065918]], [[8.959723472595215]], [[7.646790027618408]], [[7.697815418243408]], [[8.115347862243652]], [[7.619071960449219]], [[7.572359085083008]], [[7.913558006286621]], [[7.5322418212890625]], [[7.958657741546631]], [[7.420487880706787]], [[8.024356842041016]], [[8.10165023803711]], [[7.993789196014404]], [[7.642430782318115]], [[7.333017826080322]], [[7.333766937255859]], [[8.388383865356445]], [[7.282675743103027]], [[7.608148574829102]], [[8.325189590454102]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_a1f0f25338eadd41af3c9eb404a24324(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.591955184936523]], [[5.937971591949463]], [[5.838271141052246]], [[5.930294990539551]], [[5.167905807495117]], [[5.86020565032959]], [[5.7827863693237305]], [[6.046093940734863]], [[5.75948429107666]], [[5.110523223876953]], [[6.796349048614502]], [[6.009696006774902]], [[6.375565528869629]], [[6.133172512054443]], [[5.7320027351379395]], [[5.841438293457031]], [[5.657365322113037]], [[5.535276889801025]], [[5.8343400955200195]], [[5.474064826965332]], [[5.478018760681152]], [[5.710632801055908]], [[6.113531589508057]], [[6.6900506019592285]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_0b4f716f86190adc422f6186d9f1a67d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.309843063354492]], [[6.539846897125244]], [[6.768796920776367]], [[6.078126430511475]], [[7.450684070587158]], [[5.8743414878845215]], [[6.8607683181762695]], [[6.394069194793701]], [[6.257390975952148]], [[6.216409683227539]], [[6.404240608215332]], [[6.725374221801758]], [[6.286440849304199]], [[6.559449195861816]], [[7.190722942352295]], [[7.1903815269470215]], [[7.360578536987305]], [[6.062586307525635]], [[6.55659294128418]], [[6.15986967086792]], [[6.108526229858398]], [[5.927225112915039]], [[6.7410383224487305]], [[6.556881427764893]], [[7.01950216293335]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_1e2eb2a83b2fe354971c389beead5956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6438512802124023]], [[3.296189785003662]], [[2.9459354877471924]], [[3.5804953575134277]], [[3.079784631729126]], [[2.939220666885376]], [[3.0548999309539795]], [[2.908855676651001]], [[3.5852274894714355]], [[3.0470340251922607]], [[3.243905544281006]], [[3.517317533493042]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


    class TestPrimitiveOp_dfc4728a121fa36592d0b7ee8fc37f46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[709.1481323242188]], [[692.7680053710938]], [[735.9580688476562]], [[664.4910278320312]], [[767.9503784179688]], [[734.5795288085938]], [[688.29736328125]], [[765.3384399414062]], [[746.7483520507812]], [[764.594482421875]], [[721.3216552734375]], [[714.5488891601562]], [[668.8396606445312]], [[734.5792846679688]], [[712.2596435546875]], [[680.5489501953125]], [[642.421875]], [[701.2931518554688]], [[649.7561645507812]], [[729.9595336914062]], [[730.0673217773438]], [[656.1763916015625]], [[681.51171875]], [[740.1054077148438]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_df1ae749c99ea10b271d24d292854200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[88.58912658691406]], [[85.60018920898438]], [[74.9617919921875]], [[79.29644775390625]], [[82.23506927490234]], [[86.25605773925781]], [[86.15576171875]], [[92.4328842163086]], [[82.28874206542969]], [[89.9693603515625]], [[81.6507339477539]], [[84.16071319580078]], [[71.01463317871094]], [[85.02629089355469]], [[81.29607391357422]], [[79.44832611083984]], [[79.5385513305664]], [[87.98348236083984]], [[83.3065414428711]], [[81.68899536132812]], [[73.86360931396484]], [[76.48143005371094]], [[71.70701599121094]], [[85.40310668945312]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_ccee83dcafb87efe3e4d9885455aea5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45.137916564941406]], [[44.4119987487793]], [[37.813167572021484]], [[35.5390625]], [[45.238399505615234]], [[44.228973388671875]], [[43.900428771972656]], [[49.975494384765625]], [[42.41387176513672]], [[41.04962921142578]], [[38.83075714111328]], [[42.41850662231445]], [[39.964046478271484]], [[47.64216232299805]], [[47.3900260925293]], [[46.39550018310547]], [[43.45512771606445]], [[44.626033782958984]], [[42.45619201660156]], [[41.56513595581055]], [[45.767799377441406]], [[46.025306701660156]], [[41.83007049560547]], [[42.63505935668945]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_49b5e29decca78e5615e4aa5a6ab32b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[23.270776748657227]], [[19.274690628051758]], [[19.025012969970703]], [[20.659255981445312]], [[21.67249870300293]], [[22.679594039916992]], [[22.12632179260254]], [[19.17441749572754]], [[20.16863250732422]], [[21.49508285522461]], [[22.29465103149414]], [[22.043739318847656]], [[21.93906593322754]], [[20.350486755371094]], [[21.37721824645996]], [[22.92276382446289]], [[21.07513427734375]], [[23.164953231811523]], [[20.970190048217773]], [[23.452327728271484]], [[20.533618927001953]], [[21.655485153198242]], [[19.653417587280273]], [[21.133453369140625]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_450435c5e73c8942341a4ae22905db45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34209.76953125]], [[35293.09375]], [[36197.36328125]], [[41263.84765625]], [[33075.9453125]], [[39370.27734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_614ebf7809d9fe929c3f200471c854cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43286.53515625]], [[49013.6796875]], [[40533.49609375]], [[43197.0859375]], [[38508.86328125]], [[35771.30078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_441a01800f383bbe3a137c4d789bfcd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37526.0234375]], [[36022.96875]], [[39461.9140625]], [[41080.35546875]], [[40742.5625]], [[35906.984375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_16fd5f5d9eee8827d5d94549ed5411e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[40397.62109375]], [[37758.46875]], [[48596.2578125]], [[39773.49609375]], [[43267.83203125]], [[38806.35546875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


    class TestPrimitiveOp_b02f14d137834cd66cd62a33cc0b1c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.869830131530762]], [[6.091160297393799]], [[6.01346492767334]], [[5.536749362945557]], [[6.523960113525391]], [[6.541598320007324]], [[6.203537464141846]], [[5.59697961807251]], [[5.732237815856934]], [[6.437671184539795]], [[5.600796222686768]], [[5.776000499725342]], [[6.717878818511963]], [[5.755391597747803]], [[6.647819519042969]], [[5.8235297203063965]], [[5.793607234954834]], [[6.186882495880127]], [[6.169209957122803]], [[5.716612815856934]], [[6.256058216094971]], [[5.74980354309082]], [[6.225977897644043]], [[5.876616477966309]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_c1892fd06bc28863abe0b084ee530ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.33536958694458, 5.728888511657715, 4.716067790985107, 4.229830265045166, 5.404992580413818, 4.498203754425049, 5.148319721221924, 4.750048637390137, 5.1595282554626465, 4.218147277832031, 5.205923080444336, 4.9782233238220215, 4.736413478851318, 4.962367057800293, 5.054198265075684, 5.145442008972168, 4.78067684173584, 4.842703819274902]], dtype='float32').reshape([1, 18]),
            ]


    class TestPrimitiveOp_9f7e661b31e4dec2db28ebc5383946c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.162670612335205, 4.743340492248535, 5.998516082763672, 5.040639877319336, 5.3741350173950195, 5.7692341804504395, 5.374410152435303, 5.690393447875977, 6.033936023712158, 5.206454277038574, 5.5260396003723145, 5.6504645347595215, 5.447342395782471, 5.529510974884033, 5.592252254486084, 6.441046714782715, 5.761960983276367, 5.791680812835693, 5.865000247955322, 5.504603385925293, 5.820616245269775, 5.220635890960693, 5.280458450317383]], dtype='float32').reshape([1, 23]),
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


    class TestPrimitiveOp_c8b8fa3356e3b80b5ff01ce01e919562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.156549453735352]], [[6.578279972076416]], [[8.965864181518555]], [[7.614697456359863]], [[7.707426071166992]], [[7.436953067779541]], [[7.509988307952881]], [[7.788017749786377]], [[6.393144607543945]], [[7.362917900085449]], [[7.2338972091674805]], [[8.074005126953125]], [[7.098735809326172]], [[6.910137176513672]], [[8.077880859375]], [[8.106316566467285]], [[7.840653419494629]], [[8.020371437072754]], [[8.272087097167969]], [[8.352813720703125]], [[7.642253875732422]], [[7.687630653381348]], [[7.613253593444824]], [[7.520717620849609]], [[7.637831687927246]], [[8.50828742980957]], [[6.867826461791992]], [[7.150737762451172]], [[7.4321746826171875]], [[7.626999378204346]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_07830e329a61bf3e1f87cacc749af3b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.764479637145996]], [[8.717795372009277]], [[8.243857383728027]], [[6.911282062530518]], [[7.926513671875]], [[8.030611991882324]], [[8.945138931274414]], [[7.763011932373047]], [[7.695864200592041]], [[7.878115177154541]], [[8.235343933105469]], [[6.995920658111572]], [[8.233352661132812]], [[7.727132797241211]], [[7.220941066741943]], [[8.071619033813477]], [[8.144110679626465]], [[6.693134307861328]], [[7.199584484100342]], [[8.235687255859375]], [[7.900215148925781]], [[7.703789234161377]], [[7.636146545410156]], [[8.11130428314209]], [[8.535118103027344]], [[7.386641502380371]], [[8.645404815673828]], [[6.932118892669678]], [[7.863061904907227]], [[8.122811317443848]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_e100b13dd610a7f8edd1473fbce2a8a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a7fee38f0d8f89f4d59c7e49caba9ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8566051721572876]], [[2.095454216003418]], [[1.8238812685012817]], [[1.474919080734253]], [[1.5276159048080444]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_411eaeb25f1402e68712991ed341dc37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.719106674194336]], [[2.747519016265869]], [[2.2771918773651123]], [[2.710972547531128]], [[2.6780829429626465]], [[2.6255321502685547]], [[2.8236663341522217]], [[2.8805809020996094]], [[2.647463321685791]], [[2.6628236770629883]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c74adc61341ee4b217146d0e14cc4f21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.086631774902344]], [[7.308384895324707]], [[5.830771446228027]], [[6.009194850921631]], [[5.87945032119751]], [[6.114314079284668]], [[5.7845635414123535]], [[6.975741386413574]], [[5.626193523406982]], [[6.476055145263672]], [[6.509440898895264]], [[6.814225673675537]], [[6.179939270019531]], [[6.212969779968262]], [[5.426244735717773]], [[5.825172424316406]], [[6.444512367248535]], [[6.775128364562988]], [[5.420651435852051]], [[6.370400428771973]], [[6.124203205108643]], [[6.66068172454834]], [[6.493947505950928]], [[6.2271623611450195]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_a24e081a01b84edadee02dc43a06b3f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.429371356964111]], [[5.102461814880371]], [[5.122546195983887]], [[4.959865093231201]], [[4.487847328186035]], [[4.595395565032959]], [[4.401482582092285]], [[5.045962333679199]], [[5.277590274810791]], [[4.407054424285889]], [[4.078505992889404]], [[4.891146183013916]], [[4.863668918609619]], [[4.882938385009766]], [[4.876100540161133]], [[4.58686637878418]], [[4.975370407104492]], [[5.126645565032959]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35ad88c887b120e2ce63900c5c954d89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.2779717445373535]], [[6.342955589294434]], [[6.889688014984131]], [[6.494229316711426]], [[6.744277000427246]], [[6.069737911224365]], [[7.056484222412109]], [[5.741796493530273]], [[6.317704677581787]], [[6.743081569671631]], [[6.25610876083374]], [[6.39182186126709]], [[5.538364410400391]], [[6.538943290710449]], [[6.79679536819458]], [[7.517889022827148]], [[6.390634536743164]], [[6.382289886474609]], [[6.212860584259033]], [[6.586852073669434]], [[5.402479648590088]], [[6.590733051300049]], [[6.075348854064941]], [[5.696362495422363]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_eff5bfbf33e21cd79761cdd2cc3b4065(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5981800556182861]], [[1.4517943859100342]], [[0.8325352668762207]], [[1.1580177545547485]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_661b800fa7b4dcf6c02d320a8849130c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e86954e09a1f6907cc2c742f79bdf377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.759239673614502]], [[3.067654848098755]], [[2.8620004653930664]], [[2.730835437774658]], [[2.2491555213928223]], [[2.969370126724243]], [[3.7615504264831543]], [[2.9485251903533936]], [[2.366492748260498]], [[3.176340103149414]], [[2.972701072692871]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_d40ee204367ff373b3c1801203fcf119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[9.219853401184082]], [[7.977720737457275]], [[7.273802280426025]], [[8.22716236114502]], [[8.4916410446167]], [[9.26848316192627]], [[7.195425987243652]], [[7.818164825439453]], [[8.419690132141113]], [[7.483343601226807]], [[8.165380477905273]], [[7.699680328369141]], [[8.12510871887207]], [[8.931337356567383]], [[7.405519485473633]], [[9.185028076171875]], [[7.8361005783081055]], [[7.829963684082031]], [[8.57872200012207]], [[7.620147705078125]], [[8.942919731140137]], [[7.511859893798828]], [[7.994983196258545]], [[7.808498859405518]], [[8.88344955444336]], [[7.918191432952881]], [[8.144401550292969]], [[8.468025207519531]], [[8.549529075622559]], [[8.860572814941406]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_1325b489f2ea1fa4fb8dc4b7fbf9cf8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.339223384857178]], [[4.631780624389648]], [[4.155935764312744]], [[3.9742817878723145]], [[5.302306652069092]], [[4.456806659698486]], [[4.654520034790039]], [[4.203819751739502]], [[4.233545303344727]], [[3.887152910232544]], [[4.019169807434082]], [[4.785059452056885]], [[5.198037147521973]], [[4.3127031326293945]], [[3.808553695678711]], [[4.056495666503906]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_8f2edc55882188b28dcb4d79a088e84a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.9523186683654785]], [[7.42242431640625]], [[8.545364379882812]], [[8.203980445861816]], [[7.4313483238220215]], [[7.6026763916015625]], [[8.048667907714844]], [[8.16807746887207]], [[7.477578163146973]], [[7.682559967041016]], [[7.5439887046813965]], [[8.519378662109375]], [[8.315771102905273]], [[7.916327476501465]], [[7.257624626159668]], [[8.291592597961426]], [[7.157421588897705]], [[7.265719890594482]], [[7.141048908233643]], [[7.891236305236816]], [[8.022711753845215]], [[8.24140453338623]], [[7.856504917144775]], [[8.895251274108887]], [[8.04963493347168]], [[8.79507064819336]], [[8.328497886657715]], [[7.987967491149902]], [[8.476158142089844]], [[7.17471170425415]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_c381478645c9fa327b3ac237ecf59a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.717169761657715]], [[6.377462387084961]], [[6.11559534072876]], [[6.248219966888428]], [[6.040029048919678]], [[7.45786714553833]], [[7.106032371520996]], [[5.864388465881348]], [[6.851099014282227]], [[6.623362064361572]], [[6.710056781768799]], [[5.612614154815674]], [[7.330811023712158]], [[5.56170129776001]], [[7.341120719909668]], [[6.5250678062438965]], [[6.4778265953063965]], [[6.345496654510498]], [[7.3401031494140625]], [[6.775630474090576]], [[6.1982197761535645]], [[5.557824611663818]], [[6.792047023773193]], [[6.723367214202881]], [[6.0354838371276855]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_b6c35114fd821b658637923fb3c94d59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.431391716003418]], [[4.728453636169434]], [[3.8485469818115234]], [[3.7370498180389404]], [[4.240548133850098]], [[3.7491250038146973]], [[4.5072808265686035]], [[4.023393154144287]], [[3.9357423782348633]], [[4.149059295654297]], [[4.889106273651123]], [[4.266361236572266]], [[4.887792587280273]], [[4.657820701599121]], [[4.33989143371582]], [[4.628081321716309]], [[3.937661647796631]], [[4.499518394470215]], [[4.870829105377197]], [[4.252282619476318]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_06f03248cda7183c3557751986ceddef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.285511016845703]], [[4.616420745849609]], [[5.1114630699157715]], [[4.157078742980957]], [[4.800133228302002]], [[3.777223587036133]], [[4.694716930389404]], [[3.983590841293335]], [[3.9912760257720947]], [[4.235405921936035]], [[4.37108039855957]], [[4.672853469848633]], [[4.280800819396973]], [[4.807054042816162]], [[4.614780426025391]], [[4.6422810554504395]], [[4.211650371551514]], [[5.195508003234863]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_69f8944643d16804feb06a9df5d5c502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.78338623046875]], [[4.758307456970215]], [[4.5570759773254395]], [[4.651852607727051]], [[4.396996021270752]], [[4.2491278648376465]], [[4.278838634490967]], [[4.917035102844238]], [[4.2762885093688965]], [[4.933725833892822]], [[5.054064750671387]], [[4.185392379760742]], [[4.766103267669678]], [[4.256556510925293]], [[4.734835147857666]], [[5.001825332641602]], [[3.9590556621551514]], [[4.636456489562988]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_985f2e0fdb496dd46b3239326dfa0c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27aab4cc53db7b1c512a930b85c0a2f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.475754737854004]], [[5.490368366241455]], [[5.066490650177002]], [[6.535336494445801]], [[6.574573993682861]], [[6.074638843536377]], [[5.812880039215088]], [[6.25173282623291]], [[6.593845367431641]], [[6.095102787017822]], [[6.271059036254883]], [[5.55356502532959]], [[6.959198951721191]], [[5.8360443115234375]], [[6.394191741943359]], [[5.082563400268555]], [[5.476058006286621]], [[5.893261909484863]], [[5.3520402908325195]], [[6.88227653503418]], [[6.345145225524902]], [[5.516343116760254]], [[6.1561150550842285]], [[6.437520980834961]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_030909805c34712c06a25bccb75b4358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcca2e4230819f5d89bea869b722957a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.772536277770996]], [[5.173009395599365]], [[3.8997905254364014]], [[5.0453996658325195]], [[4.3154191970825195]], [[5.118714809417725]], [[4.704500198364258]], [[4.647329807281494]], [[4.215498924255371]], [[4.829869270324707]], [[4.632633686065674]], [[5.515543460845947]], [[4.459994792938232]], [[4.442266464233398]], [[4.744085788726807]], [[4.792042255401611]], [[4.764348030090332]], [[4.661513805389404]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_4ad1db52585bfa97bcfbf9bab07d1a74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.520150184631348]], [[4.817781925201416]], [[4.683832168579102]], [[5.0453901290893555]], [[4.88814115524292]], [[3.796255111694336]], [[5.0784525871276855]], [[4.307088375091553]], [[4.392723560333252]], [[5.034638404846191]], [[4.890681266784668]], [[4.789068222045898]], [[4.620427131652832]], [[4.711824893951416]], [[4.296992778778076]], [[4.6062116622924805]], [[4.63204288482666]], [[5.542410373687744]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_8590ab468d89148794534d699a7c0e7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.531711101531982]], [[4.344454288482666]], [[4.5150837898254395]], [[4.630534648895264]], [[4.285594463348389]], [[4.032207489013672]], [[3.536329746246338]], [[4.6597137451171875]], [[4.211676597595215]], [[4.191618919372559]], [[4.450105667114258]], [[5.07877779006958]], [[4.409242153167725]], [[4.230588436126709]], [[4.868673324584961]], [[4.5204758644104]], [[5.747686386108398]], [[4.947834491729736]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_cd86624b2fa2fe9dcdc297d0310116a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.25018835067749]], [[4.311614513397217]], [[4.094573497772217]], [[3.818601131439209]], [[3.6546530723571777]], [[3.475036144256592]], [[3.1861672401428223]], [[3.8367486000061035]], [[4.038932800292969]], [[4.63492488861084]], [[3.6220250129699707]], [[3.9613070487976074]], [[3.4617204666137695]], [[4.631092548370361]], [[3.767266273498535]], [[4.41674280166626]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_5b8b1516851ca68d1551bf1cd343b97e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.675265789031982]], [[4.467334270477295]], [[3.612835645675659]], [[4.771159648895264]], [[4.658884525299072]], [[4.322079658508301]], [[4.130802631378174]], [[4.945823669433594]], [[4.1770548820495605]], [[4.741894245147705]], [[4.489341735839844]], [[4.604240417480469]], [[4.373265266418457]], [[3.9902279376983643]], [[4.216479778289795]], [[4.061892986297607]], [[4.76384973526001]], [[3.904416561126709]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_1f2f439c117ef471745391bfe0395156(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9944493174552917]], [[1.2166593074798584]], [[1.1984256505966187]], [[1.1523396968841553]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_54a4370d8a9faed732201b42115f5c62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.8062310218811035]], [[5.361790180206299]], [[4.5029802322387695]], [[4.500344753265381]], [[4.982926845550537]], [[4.848570346832275]], [[4.778101921081543]], [[5.076159954071045]], [[4.993417263031006]], [[4.963909149169922]], [[4.629103660583496]], [[4.729391098022461]], [[5.710188865661621]], [[4.812963008880615]], [[5.511858940124512]], [[5.036686420440674]], [[5.425128936767578]], [[4.4527363777160645]], [[5.328535556793213]], [[5.170138359069824]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_e341740e0f212bca7cbc0355e70c8765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_212cd5ad0ef81ca57ac8b97b66e02890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.342505693435669]], [[2.7520391941070557]], [[2.714287519454956]], [[3.211639881134033]], [[3.0043015480041504]], [[3.492180347442627]], [[3.593163013458252]], [[3.679666042327881]], [[2.5081582069396973]], [[3.2115612030029297]], [[3.288628101348877]], [[3.0804831981658936]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_ced7988877fb7eb1323ef999ab600570(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.7912914752960205]], [[4.724730014801025]], [[5.298625946044922]], [[5.1255106925964355]], [[4.62327241897583]], [[5.113589286804199]], [[5.026547431945801]], [[4.239257335662842]], [[4.4622602462768555]], [[5.033857822418213]], [[3.9976367950439453]], [[4.5428996086120605]], [[4.076603889465332]], [[4.696175575256348]], [[4.704270839691162]], [[4.375837326049805]], [[4.985689640045166]], [[4.377310276031494]], [[4.806520938873291]], [[4.063906669616699]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_b7ef5127fac19f962d162beb340bf2a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.0060856342315674]], [[2.823049545288086]], [[2.855907440185547]], [[3.3607177734375]], [[3.1934914588928223]], [[3.269962787628174]], [[2.899298667907715]], [[3.100433111190796]], [[3.093851089477539]], [[2.913404703140259]], [[2.969477653503418]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_3d8463bf9c558987eebb0fa9682df449(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.2398808002471924]], [[3.508150577545166]], [[2.8001081943511963]], [[3.5751781463623047]], [[2.6233019828796387]], [[3.1378047466278076]], [[3.546370506286621]], [[3.4879634380340576]], [[3.134951114654541]], [[3.1070199012756348]], [[3.4020214080810547]], [[2.941101312637329]], [[3.3202311992645264]], [[3.3293280601501465]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


    class TestPrimitiveOp_89425906fb09e236d5a82fcc18553252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.995167255401611]], [[5.068471908569336]], [[5.232618808746338]], [[4.3133463859558105]], [[5.070679664611816]], [[4.943248271942139]], [[5.22681188583374]], [[4.9932475090026855]], [[5.181914806365967]], [[4.9883599281311035]], [[4.587845325469971]], [[4.951524257659912]], [[4.5863518714904785]], [[4.5127482414245605]], [[4.808051109313965]], [[5.10516881942749]], [[4.906620979309082]], [[4.632011413574219]], [[4.7547783851623535]], [[4.731569766998291]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_d8731afabd1469fad2bc27fcdb1c719c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43951.48828125]], [[34423.453125]], [[34047.71484375]], [[29783.375]], [[37019.40234375]], [[38497.50390625]]], [[[43313.31640625]], [[33921.72265625]], [[33559.36328125]], [[29355.10546875]], [[36486.9140625]], [[37942.50390625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_b8bc2b0389c6d5bf9b9ccf64055c1a0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37845.80859375]], [[32601.22265625]], [[41104.3828125]], [[33579.23828125]], [[41183.9921875]], [[40182.13671875]]], [[[36832.75390625]], [[31724.77734375]], [[40006.65625]], [[32682.271484375]], [[40081.921875]], [[39108.6875]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_283e449165f411dd6c5f1ae80a2646ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[47123.75]], [[41793.8828125]], [[49294.4765625]], [[44015.36328125]], [[40978.10546875]], [[35567.5703125]]], [[[45730.7265625]], [[40560.58203125]], [[47835.48828125]], [[42716.0703125]], [[39762.51953125]], [[34511.59375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_ae14ef26a56438fa3fc36d88709a0caf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37736.85546875]], [[39921.3515625]], [[44999.49609375]], [[36545.47265625]], [[40640.953125]], [[44243.4921875]]], [[[36904.62109375]], [[39050.265625]], [[44017.51171875]], [[35748.12890625]], [[39746.23828125]], [[43278.5859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


    class TestPrimitiveOp_56527d75752fc84c4b9927a226af1e54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.301014423370361]], [[8.403518676757812]], [[8.03527545928955]], [[8.760984420776367]], [[8.188155174255371]], [[7.942190170288086]], [[7.466071128845215]], [[7.920556545257568]], [[8.333847045898438]], [[8.276917457580566]], [[6.924700736999512]], [[7.403339385986328]], [[7.732781410217285]], [[8.625016212463379]], [[6.764622688293457]], [[7.116641044616699]], [[8.37065601348877]], [[8.823826789855957]], [[8.26421070098877]], [[8.392292022705078]], [[6.971035480499268]], [[7.877248287200928]], [[8.675994873046875]], [[7.90565299987793]], [[8.166635513305664]], [[8.00434684753418]], [[7.3291497230529785]], [[7.97578239440918]], [[7.904312610626221]], [[8.364042282104492]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_ee159955ed2c126a9185be577f8b3aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.341287136077881]], [[7.299825191497803]], [[8.345277786254883]], [[7.708090782165527]], [[6.989670753479004]], [[7.035129070281982]], [[7.876548767089844]], [[7.641441822052002]], [[7.257607460021973]], [[7.430710792541504]], [[7.43454647064209]], [[7.339380741119385]], [[7.949814319610596]], [[7.445698261260986]], [[7.395953178405762]], [[7.797006130218506]], [[7.37500524520874]], [[7.733424186706543]], [[7.49075984954834]], [[7.240258693695068]], [[7.815331935882568]], [[8.243738174438477]], [[7.368047714233398]], [[7.8810553550720215]], [[8.4630126953125]], [[7.198576927185059]], [[6.862584114074707]], [[8.420339584350586]], [[7.592401504516602]], [[7.650505542755127]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_b5b16d26e52c291dbc628fe7ba3e2eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc88004cd5fd39e8ce99afaec9070b96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.334691047668457]], [[6.553110122680664]], [[7.550365924835205]], [[7.175827980041504]], [[7.56691837310791]], [[7.009180545806885]], [[7.625830173492432]], [[6.939151763916016]], [[7.596673488616943]], [[6.650327682495117]], [[7.773587703704834]], [[7.105720043182373]], [[7.4982757568359375]], [[6.71769905090332]], [[7.4405083656311035]], [[7.593581199645996]], [[7.8551530838012695]], [[7.435949802398682]], [[8.307052612304688]], [[7.227053642272949]], [[7.727344036102295]], [[7.8308796882629395]], [[7.484035968780518]], [[6.645956516265869]], [[7.704050064086914]], [[7.600241184234619]], [[6.777804851531982]], [[7.476434230804443]], [[7.012086868286133]], [[7.311110496520996]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_fc5ef1379947ba0bbf9234d2f82db648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.924079895019531]], [[7.3153605461120605]], [[7.995610237121582]], [[7.2530198097229]], [[7.813193321228027]], [[7.922425270080566]], [[7.419620990753174]], [[7.603722095489502]], [[7.701774597167969]], [[7.354045867919922]], [[7.252942085266113]], [[8.358896255493164]], [[7.384546756744385]], [[8.001121520996094]], [[7.920747756958008]], [[7.5409440994262695]], [[7.896615505218506]], [[8.030363082885742]], [[8.464024543762207]], [[8.035039901733398]], [[7.323920726776123]], [[7.297919750213623]], [[7.611123085021973]], [[7.6905646324157715]], [[6.916043281555176]], [[6.890603542327881]], [[7.488044261932373]], [[6.946822166442871]], [[7.933925151824951]], [[7.751521587371826]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_cb6bdfdde317172b4c1f522713818e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.44456148147583]], [[3.229776382446289]], [[3.124418258666992]], [[3.338461399078369]], [[3.577549457550049]], [[3.919827699661255]], [[3.9549551010131836]], [[3.856282949447632]], [[4.125796318054199]], [[3.8217051029205322]], [[2.8720881938934326]], [[3.986161947250366]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_bf3077b508f3a81e7859e971249a0e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.021803140640259]], [[2.8979740142822266]], [[2.6711370944976807]], [[3.363351345062256]], [[2.642312526702881]], [[3.2148635387420654]], [[3.0581884384155273]], [[3.5348715782165527]], [[3.4332950115203857]], [[3.799393653869629]], [[3.1188201904296875]], [[2.997915029525757]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_d096938fa03857d1e523e861c5b2d0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.129674911499023]], [[6.374298095703125]], [[6.090598106384277]], [[6.553129196166992]], [[6.5286946296691895]], [[6.757339000701904]], [[6.135092735290527]], [[6.8264689445495605]], [[6.315954685211182]], [[7.21508264541626]], [[6.141348838806152]], [[5.7715253829956055]], [[6.627416610717773]], [[6.3487701416015625]], [[6.562192916870117]], [[6.3334736824035645]], [[6.540079116821289]], [[6.168745517730713]], [[6.445700645446777]], [[6.114558696746826]], [[5.674906253814697]], [[6.7818779945373535]], [[6.279696941375732]], [[6.37258243560791]], [[6.395266532897949]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_0cbc3973cd2587bfb50764e11192ca01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.234936237335205]], [[4.312356948852539]], [[4.602973937988281]], [[4.165308952331543]], [[4.123189926147461]], [[4.576277732849121]], [[4.517114639282227]], [[4.990133285522461]], [[4.928924083709717]], [[4.565418720245361]], [[4.6072797775268555]], [[4.104726791381836]], [[4.53456974029541]], [[4.675724983215332]], [[5.090818405151367]], [[4.430437088012695]], [[4.960774898529053]], [[4.91608190536499]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_6480f898595034dc31697d642337f1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b6d42584c6b44274a50b061a8c3be8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.412215232849121]], [[1.8416154384613037]], [[1.87169349193573]], [[1.7094300985336304]], [[1.4866801500320435]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_0dd35991d3bd02eef52c0cde33bd0f28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6466522216796875]], [[3.1592297554016113]], [[2.6499240398406982]], [[3.1042873859405518]], [[2.4753024578094482]], [[2.788867950439453]], [[3.2296197414398193]], [[2.8998422622680664]], [[2.356828212738037]], [[2.5325305461883545]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_4fe9a2507e7bc2477c92aeb3f270fcd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.219168663024902]], [[4.386056900024414]], [[4.218252658843994]], [[3.8906869888305664]], [[3.821678876876831]], [[5.3758955001831055]], [[4.510594844818115]], [[4.741876125335693]], [[4.260367393493652]], [[3.830246686935425]], [[4.32592248916626]], [[4.207342147827148]], [[4.228832244873047]], [[4.594235420227051]], [[4.741360664367676]], [[4.85493803024292]], [[4.056390762329102]], [[4.101260185241699]], [[5.8075480461120605]], [[4.5099711418151855]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_5bdcfee0b99248903605883900affff0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.851374626159668]], [[6.279842853546143]], [[6.162980556488037]], [[6.155672550201416]], [[6.570970058441162]], [[6.711134910583496]], [[6.90558385848999]], [[6.065819263458252]], [[7.283596992492676]], [[7.3350701332092285]], [[7.430379867553711]], [[6.806755542755127]], [[6.66092586517334]], [[6.8292365074157715]], [[7.115354061126709]], [[7.544079303741455]], [[6.50487756729126]], [[6.814257621765137]], [[6.6645402908325195]], [[6.562115669250488]], [[7.111221790313721]], [[6.755922794342041]], [[7.447986602783203]], [[6.774875164031982]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_2fe93505efa16cf7c1067f074d59de7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1648051430d091f6bfb1ebebdda055ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.566150188446045]], [[2.799795389175415]], [[2.5570602416992188]], [[2.6287052631378174]], [[2.6062774658203125]], [[2.6669459342956543]], [[2.548785924911499]], [[3.1859357357025146]], [[2.9422526359558105]], [[2.615391254425049]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_c7fa8edf8e02b89e20b75d856befceb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.051373481750488]], [[4.929189682006836]], [[4.523844242095947]], [[5.058758735656738]], [[4.941986560821533]], [[4.987974166870117]], [[4.642576694488525]], [[4.768600940704346]], [[4.74896240234375]], [[4.5014119148254395]], [[5.0258355140686035]], [[5.735795021057129]], [[5.5453782081604]], [[4.34968376159668]], [[4.784318923950195]], [[5.109044075012207]], [[5.173520088195801]], [[5.17549467086792]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_aff3963c390ea1ac26aae980fa7fb645(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.930376052856445, 7.958884239196777, 8.756532669067383, 8.268275260925293, 8.45052719116211, 7.764630317687988, 7.969349384307861, 8.35740852355957, 9.029668807983398, 9.289926528930664, 8.03172492980957, 8.313129425048828, 8.658809661865234, 8.234850883483887, 9.34371280670166, 8.602219581604004, 8.031670570373535, 8.66115951538086, 8.051887512207031, 9.021086692810059, 7.82981014251709, 8.385726928710938, 7.536729335784912, 8.938192367553711, 7.878767013549805, 9.028979301452637, 9.251540184020996, 8.633338928222656, 8.135558128356934, 7.85642671585083]], dtype='float32').reshape([1, 30]),
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


    class TestPrimitiveOp_a69936da4a8c4d0bbdb659f2128decea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.967134952545166]], [[7.665541172027588]], [[7.688358783721924]], [[8.009965896606445]], [[8.143436431884766]], [[8.406558990478516]], [[9.000687599182129]], [[7.82932710647583]], [[7.898717403411865]], [[7.775198936462402]], [[7.320363998413086]], [[6.782022476196289]], [[7.875368118286133]], [[8.605768203735352]], [[8.65597152709961]], [[7.1342549324035645]], [[7.60582160949707]], [[7.848722457885742]], [[8.213285446166992]], [[8.337427139282227]], [[7.865445137023926]], [[7.648224830627441]], [[8.587637901306152]], [[7.777416706085205]], [[8.325520515441895]], [[8.016615867614746]], [[8.52187442779541]], [[7.837628364562988]], [[8.583416938781738]], [[7.866849899291992]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_110e28843d10bfe5cfc6d86dcda1c490(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8248137831687927]], [[1.2744824886322021]], [[1.3420348167419434]], [[1.3940320014953613]], [[1.0370404720306396]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_68abd32f2148fa2c7de968a1f1c95493(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.8600144386291504]], [[2.6740565299987793]], [[2.925100088119507]], [[2.906919002532959]], [[2.801386833190918]], [[2.7032203674316406]], [[3.2327170372009277]], [[1.9958714246749878]], [[2.505014181137085]], [[2.3930094242095947]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_8f3538c0bd4829dc74903632d9eb1aaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.5436859130859375]], [[5.52134370803833]], [[5.874619483947754]], [[5.935561180114746]], [[6.13942813873291]], [[5.741078853607178]], [[6.110844135284424]], [[5.217152118682861]], [[5.579229354858398]], [[5.655007362365723]], [[6.007051944732666]], [[5.698542594909668]], [[5.283339500427246]], [[6.308893203735352]], [[6.3184733390808105]], [[5.412240505218506]], [[5.156081199645996]], [[6.472560882568359]], [[5.587553977966309]], [[5.511021614074707]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a57bf759df136fdd4163066cb6e76be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.9886651039123535]], [[3.6906166076660156]], [[3.822950601577759]], [[3.7890625]], [[4.127457618713379]], [[3.757260322570801]], [[4.472038745880127]], [[3.595747947692871]], [[4.517457008361816]], [[3.7042181491851807]], [[3.96732234954834]], [[4.572327613830566]], [[4.259017467498779]], [[3.8095781803131104]], [[4.457819938659668]], [[3.9950358867645264]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_effab09d3ba62c644a9f82f7cf53de38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3726940155029297]], [[3.6283209323883057]], [[3.0741968154907227]], [[3.126549243927002]], [[2.7311394214630127]], [[3.133653402328491]], [[3.4546656608581543]], [[2.7111501693725586]], [[3.43560791015625]], [[3.4273197650909424]], [[3.554928779602051]], [[2.9367902278900146]], [[3.5619406700134277]], [[3.9031782150268555]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_da718217efbb7ab6a2a7243f5904be77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.941735744476318]], [[5.350258827209473]], [[4.993419170379639]], [[4.997262954711914]], [[4.465219020843506]], [[5.24635648727417]], [[4.697439193725586]], [[4.72060489654541]], [[4.484288215637207]], [[4.6957550048828125]], [[4.52725076675415]], [[5.010762691497803]], [[4.304614543914795]], [[4.8619818687438965]], [[4.338233470916748]], [[4.847714900970459]], [[4.7345099449157715]], [[5.4691667556762695]], [[5.441762924194336]], [[4.312991619110107]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_28f46023d6121f24c5f66f7c1899588e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.716361999511719]], [[8.297204971313477]], [[7.497693061828613]], [[7.400525093078613]], [[8.344557762145996]], [[8.721635818481445]], [[7.63230037689209]], [[7.712268829345703]], [[8.096673965454102]], [[7.462275505065918]], [[8.959723472595215]], [[7.646790027618408]], [[7.697815418243408]], [[8.115347862243652]], [[7.619071960449219]], [[7.572359085083008]], [[7.913558006286621]], [[7.5322418212890625]], [[7.958657741546631]], [[7.420487880706787]], [[8.024356842041016]], [[8.10165023803711]], [[7.993789196014404]], [[7.642430782318115]], [[7.333017826080322]], [[7.333766937255859]], [[8.388383865356445]], [[7.282675743103027]], [[7.608148574829102]], [[8.325189590454102]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_5c98ea9d89d3c017ee5d6ff10019834b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.591955184936523]], [[5.937971591949463]], [[5.838271141052246]], [[5.930294990539551]], [[5.167905807495117]], [[5.86020565032959]], [[5.7827863693237305]], [[6.046093940734863]], [[5.75948429107666]], [[5.110523223876953]], [[6.796349048614502]], [[6.009696006774902]], [[6.375565528869629]], [[6.133172512054443]], [[5.7320027351379395]], [[5.841438293457031]], [[5.657365322113037]], [[5.535276889801025]], [[5.8343400955200195]], [[5.474064826965332]], [[5.478018760681152]], [[5.710632801055908]], [[6.113531589508057]], [[6.6900506019592285]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_bcbd6dbc0eb84247d5d4404bc31f165c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.309843063354492]], [[6.539846897125244]], [[6.768796920776367]], [[6.078126430511475]], [[7.450684070587158]], [[5.8743414878845215]], [[6.8607683181762695]], [[6.394069194793701]], [[6.257390975952148]], [[6.216409683227539]], [[6.404240608215332]], [[6.725374221801758]], [[6.286440849304199]], [[6.559449195861816]], [[7.190722942352295]], [[7.1903815269470215]], [[7.360578536987305]], [[6.062586307525635]], [[6.55659294128418]], [[6.15986967086792]], [[6.108526229858398]], [[5.927225112915039]], [[6.7410383224487305]], [[6.556881427764893]], [[7.01950216293335]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_e7a7b2adfb6b7648dcdf29732153662d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.6438512802124023]], [[3.296189785003662]], [[2.9459354877471924]], [[3.5804953575134277]], [[3.079784631729126]], [[2.939220666885376]], [[3.0548999309539795]], [[2.908855676651001]], [[3.5852274894714355]], [[3.0470340251922607]], [[3.243905544281006]], [[3.517317533493042]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


    class TestPrimitiveOp_2b77133d578960a675410077932ea470(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[709.1481323242188]], [[692.7680053710938]], [[735.9580688476562]], [[664.4910278320312]], [[767.9503784179688]], [[734.5795288085938]], [[688.29736328125]], [[765.3384399414062]], [[746.7483520507812]], [[764.594482421875]], [[721.3216552734375]], [[714.5488891601562]], [[668.8396606445312]], [[734.5792846679688]], [[712.2596435546875]], [[680.5489501953125]], [[642.421875]], [[701.2931518554688]], [[649.7561645507812]], [[729.9595336914062]], [[730.0673217773438]], [[656.1763916015625]], [[681.51171875]], [[740.1054077148438]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_91e1b49192d28d53b4e0f9b07e083317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[88.58912658691406]], [[85.60018920898438]], [[74.9617919921875]], [[79.29644775390625]], [[82.23506927490234]], [[86.25605773925781]], [[86.15576171875]], [[92.4328842163086]], [[82.28874206542969]], [[89.9693603515625]], [[81.6507339477539]], [[84.16071319580078]], [[71.01463317871094]], [[85.02629089355469]], [[81.29607391357422]], [[79.44832611083984]], [[79.5385513305664]], [[87.98348236083984]], [[83.3065414428711]], [[81.68899536132812]], [[73.86360931396484]], [[76.48143005371094]], [[71.70701599121094]], [[85.40310668945312]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f39f3e450df791eebe4c0b84eacfaed0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[45.137916564941406]], [[44.4119987487793]], [[37.813167572021484]], [[35.5390625]], [[45.238399505615234]], [[44.228973388671875]], [[43.900428771972656]], [[49.975494384765625]], [[42.41387176513672]], [[41.04962921142578]], [[38.83075714111328]], [[42.41850662231445]], [[39.964046478271484]], [[47.64216232299805]], [[47.3900260925293]], [[46.39550018310547]], [[43.45512771606445]], [[44.626033782958984]], [[42.45619201660156]], [[41.56513595581055]], [[45.767799377441406]], [[46.025306701660156]], [[41.83007049560547]], [[42.63505935668945]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_7c478cbafb909656fb9a1cf4830f5e4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[23.270776748657227]], [[19.274690628051758]], [[19.025012969970703]], [[20.659255981445312]], [[21.67249870300293]], [[22.679594039916992]], [[22.12632179260254]], [[19.17441749572754]], [[20.16863250732422]], [[21.49508285522461]], [[22.29465103149414]], [[22.043739318847656]], [[21.93906593322754]], [[20.350486755371094]], [[21.37721824645996]], [[22.92276382446289]], [[21.07513427734375]], [[23.164953231811523]], [[20.970190048217773]], [[23.452327728271484]], [[20.533618927001953]], [[21.655485153198242]], [[19.653417587280273]], [[21.133453369140625]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_3fe4905da00307bcf4870dedcad43696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34209.76953125]], [[35293.09375]], [[36197.36328125]], [[41263.84765625]], [[33075.9453125]], [[39370.27734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_2ac39f19b28ac28ffb06bf2e7b703fc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43286.53515625]], [[49013.6796875]], [[40533.49609375]], [[43197.0859375]], [[38508.86328125]], [[35771.30078125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_61db378ab6e553edfe0c33c2b8a6c903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[37526.0234375]], [[36022.96875]], [[39461.9140625]], [[41080.35546875]], [[40742.5625]], [[35906.984375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_cf1a07d0ca352dbaf1389911a0d821eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[40397.62109375]], [[37758.46875]], [[48596.2578125]], [[39773.49609375]], [[43267.83203125]], [[38806.35546875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


    class TestPrimitiveOp_1e65aed73ab0e2161fc21f61e5f6cda5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.869830131530762]], [[6.091160297393799]], [[6.01346492767334]], [[5.536749362945557]], [[6.523960113525391]], [[6.541598320007324]], [[6.203537464141846]], [[5.59697961807251]], [[5.732237815856934]], [[6.437671184539795]], [[5.600796222686768]], [[5.776000499725342]], [[6.717878818511963]], [[5.755391597747803]], [[6.647819519042969]], [[5.8235297203063965]], [[5.793607234954834]], [[6.186882495880127]], [[6.169209957122803]], [[5.716612815856934]], [[6.256058216094971]], [[5.74980354309082]], [[6.225977897644043]], [[5.876616477966309]]]], dtype='float32').reshape([1, 24, 1, 1]),
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