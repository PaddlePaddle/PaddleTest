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


    class TestPrimitiveOp_46968e42bdcf7702bfb4d0a9ede7b441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202bfdc0ac9db07356cae4b693623ccc
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.516043186187744, 4.707422256469727, 5.164888858795166, 5.325064182281494, 5.265877723693848, 5.291617393493652, 5.1853132247924805, 5.42218017578125, 4.414474010467529, 4.513939380645752, 5.1259446144104, 4.9605937004089355, 4.702442169189453, 5.391978740692139, 5.0975661277771, 5.194991588592529, 5.173627853393555, 4.850863933563232]], dtype='float32').reshape([1, 18]),
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


    class TestPrimitiveOp_78142e80797cd10ad81a7977aa23f1cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd6460d9338b9f965d298e71d4ef198a
        def get_inputs(self):
            return [
                paddle.to_tensor([[6.707153797149658, 6.260829448699951, 6.551505088806152, 6.331302642822266, 6.301877975463867, 6.822994709014893, 6.100552558898926, 6.320962905883789, 5.698636531829834, 6.4518914222717285, 5.776687145233154, 7.555084705352783, 5.768364906311035, 6.92980432510376, 6.29964542388916, 5.982891082763672, 6.051077365875244, 6.072269916534424, 5.807013034820557, 6.226430416107178, 6.443246364593506, 6.073713302612305, 6.26926326751709]], dtype='float32').reshape([1, 23]),
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


    class TestPrimitiveOp_3b57acef5f24fa49371026900880a61c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.477514266967773]], [[7.119456768035889]], [[7.085227012634277]], [[6.6396708488464355]], [[7.529614448547363]], [[7.339506149291992]], [[7.696058750152588]], [[7.32058572769165]], [[7.314259052276611]], [[7.595938205718994]], [[7.538951873779297]], [[6.933803081512451]], [[7.610673427581787]], [[7.8937482833862305]], [[7.015742301940918]], [[8.132213592529297]], [[7.123446941375732]], [[7.175824165344238]], [[7.174844741821289]], [[7.820660591125488]], [[7.858874797821045]], [[6.548339366912842]], [[7.482433319091797]], [[7.067334175109863]], [[7.9104390144348145]], [[6.727339267730713]], [[6.801524639129639]], [[7.090723037719727]], [[7.15854024887085]], [[7.843683242797852]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_42f3feb5af83bee07ab916184ea716fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.417559623718262]], [[7.743318557739258]], [[7.118884086608887]], [[7.612246036529541]], [[7.387399196624756]], [[8.145689010620117]], [[8.002328872680664]], [[7.574892520904541]], [[7.7750959396362305]], [[7.651700019836426]], [[7.327408790588379]], [[6.881934642791748]], [[8.146565437316895]], [[6.961785793304443]], [[7.403148651123047]], [[7.219143390655518]], [[7.513122081756592]], [[7.609578609466553]], [[8.403047561645508]], [[8.094966888427734]], [[7.281956672668457]], [[6.785111427307129]], [[7.188939094543457]], [[7.072811603546143]], [[7.33947229385376]], [[7.32002592086792]], [[7.809920787811279]], [[7.630326747894287]], [[7.602578163146973]], [[7.1075544357299805]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_503dba832eb14b2fbdf719e8a38912e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa6a208551763b029a4175fcd015eae
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.369239330291748]], [[1.2205345630645752]], [[1.3513569831848145]], [[1.3422176837921143]], [[1.3516736030578613]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


    class TestPrimitiveOp_c493d1e261f3f87e5b3d01df85ea1ba2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.545330047607422]], [[2.542398691177368]], [[2.236086130142212]], [[2.722609519958496]], [[2.6072356700897217]], [[3.208588123321533]], [[3.3306355476379395]], [[2.2302825450897217]], [[2.476623058319092]], [[2.767191171646118]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_403091076002ed8e0d3f571fd4ab0950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.671944618225098]], [[6.098160743713379]], [[5.53664493560791]], [[5.733607292175293]], [[6.2740702629089355]], [[5.394620895385742]], [[5.611918926239014]], [[5.351341724395752]], [[6.054460525512695]], [[5.389400959014893]], [[5.4167094230651855]], [[5.1283745765686035]], [[5.846133232116699]], [[5.851635456085205]], [[5.39577579498291]], [[6.457047462463379]], [[6.039535045623779]], [[6.229255676269531]], [[5.209201335906982]], [[5.494519233703613]], [[6.435673236846924]], [[6.3030104637146]], [[4.4522929191589355]], [[6.078214645385742]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_02e35c95034e7b82c391c33442409983(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.120477676391602]], [[4.59541130065918]], [[4.452274799346924]], [[3.8959805965423584]], [[4.421902179718018]], [[4.485260963439941]], [[5.113558769226074]], [[4.413151264190674]], [[4.051578521728516]], [[4.178587436676025]], [[4.9391093254089355]], [[4.186689853668213]], [[4.489201068878174]], [[4.360198497772217]], [[4.403806686401367]], [[4.735654354095459]], [[4.867974281311035]], [[4.216265678405762]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_25f3390a1ef88956ccefb87cfdd829e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a93d395e6896e9fdb32b92390fc5c09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb4f053c1231e2c29a9bc3a688b54cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.3108625411987305]], [[6.79969596862793]], [[6.751509666442871]], [[6.1104865074157715]], [[6.556639194488525]], [[6.563843250274658]], [[6.04982328414917]], [[6.986202716827393]], [[6.39040470123291]], [[6.591248035430908]], [[6.4414215087890625]], [[6.0124945640563965]], [[6.929512977600098]], [[6.34401798248291]], [[7.0286431312561035]], [[6.831506252288818]], [[5.958920955657959]], [[5.804043292999268]], [[5.180810928344727]], [[5.40272331237793]], [[6.816746711730957]], [[5.976834774017334]], [[6.840426445007324]], [[6.716890811920166]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_a87eea85b4de8a8369c33d8d8186824c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81236129c333dfe7ae73bbcbb0979cbf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8333555459976196]], [[0.4858071208000183]], [[1.1957030296325684]], [[1.2955039739608765]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_35dd1bb63ea759734b9e07adf2350a38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a79f19dffaf1a401b1a360fa95eb71
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5765128135681152]], [[2.6508169174194336]], [[2.5669305324554443]], [[2.265747547149658]], [[2.859575033187866]], [[3.018110752105713]], [[2.4999642372131348]], [[2.6135239601135254]], [[2.8138022422790527]], [[2.400343894958496]], [[2.293578863143921]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_5b92fee68bb9a42432e29126325d15b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.678655624389648]], [[7.421043872833252]], [[7.546217441558838]], [[7.017660140991211]], [[7.533844947814941]], [[7.912869453430176]], [[7.165874004364014]], [[7.757705211639404]], [[7.679967880249023]], [[7.868678569793701]], [[7.383896350860596]], [[7.19304895401001]], [[7.269775390625]], [[7.78610897064209]], [[7.44060754776001]], [[7.792629241943359]], [[6.972398281097412]], [[7.668750286102295]], [[7.87344217300415]], [[7.160478591918945]], [[7.612797737121582]], [[7.538360118865967]], [[7.100478172302246]], [[7.216472148895264]], [[7.498114109039307]], [[7.365005970001221]], [[7.5870442390441895]], [[7.954311847686768]], [[6.997818470001221]], [[6.68562650680542]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_557b26aa90da26347570487d282ad0ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.204582214355469]], [[4.17450475692749]], [[4.450677394866943]], [[5.001364707946777]], [[4.378679275512695]], [[4.139589786529541]], [[4.008517742156982]], [[4.55598258972168]], [[4.8348588943481445]], [[3.643404006958008]], [[5.065579891204834]], [[4.885112285614014]], [[4.120347499847412]], [[4.529646873474121]], [[3.988499879837036]], [[3.9207963943481445]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_6e9ddffc260684634992f15976119519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.332149505615234]], [[8.124646186828613]], [[9.008127212524414]], [[7.852988243103027]], [[7.852867126464844]], [[7.848668575286865]], [[8.38390064239502]], [[7.666966915130615]], [[8.083961486816406]], [[8.015180587768555]], [[8.18043327331543]], [[7.053821086883545]], [[7.988963603973389]], [[7.258120536804199]], [[7.686677932739258]], [[7.7280659675598145]], [[7.685739040374756]], [[7.41143274307251]], [[6.809802532196045]], [[8.155046463012695]], [[6.916194438934326]], [[7.524030685424805]], [[8.538736343383789]], [[7.1718244552612305]], [[7.731924057006836]], [[7.71543550491333]], [[7.578956604003906]], [[7.180920124053955]], [[7.5848236083984375]], [[7.148171424865723]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_212267eb69ad8b442ffabdaac4b9988e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.464395046234131]], [[6.970254421234131]], [[6.331833839416504]], [[6.208786487579346]], [[6.412256240844727]], [[6.44594144821167]], [[6.668659687042236]], [[5.506393909454346]], [[5.857957363128662]], [[6.159074783325195]], [[6.215778350830078]], [[7.449837684631348]], [[6.134645462036133]], [[7.1130499839782715]], [[6.447407245635986]], [[6.364443778991699]], [[6.062101364135742]], [[6.829754829406738]], [[6.665282249450684]], [[5.587331295013428]], [[6.970404624938965]], [[6.038416385650635]], [[6.559352397918701]], [[7.202384948730469]], [[6.573240756988525]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_a249b77fc501f522957bd1e5a576d7e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.648985385894775]], [[4.977217197418213]], [[4.969912052154541]], [[4.990406513214111]], [[5.3486456871032715]], [[5.269417762756348]], [[5.3413591384887695]], [[4.685579776763916]], [[4.341116905212402]], [[5.372086524963379]], [[4.675849914550781]], [[5.339875221252441]], [[4.787026405334473]], [[4.957879066467285]], [[4.992903709411621]], [[4.947808742523193]], [[4.469698905944824]], [[3.9869487285614014]], [[4.38286018371582]], [[5.121321201324463]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_0feb1ef3e9c2658a4190c9b455a452a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.745029449462891]], [[4.362461090087891]], [[4.826979160308838]], [[4.8406877517700195]], [[4.93057918548584]], [[4.475962162017822]], [[4.890925407409668]], [[4.360519886016846]], [[5.017579555511475]], [[5.275971412658691]], [[4.956449508666992]], [[4.257177829742432]], [[4.3468146324157715]], [[4.757590293884277]], [[4.607741355895996]], [[4.7346930503845215]], [[5.006930828094482]], [[5.1289238929748535]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_efa814637ff6d0393cae81020f40ed69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.018347263336182]], [[5.169200897216797]], [[4.998176097869873]], [[4.5034708976745605]], [[4.442930221557617]], [[5.493774890899658]], [[5.000889301300049]], [[4.583656311035156]], [[5.095301628112793]], [[5.118431568145752]], [[4.825158596038818]], [[4.5962653160095215]], [[4.8283796310424805]], [[4.349386215209961]], [[4.9702229499816895]], [[4.948547840118408]], [[4.366069793701172]], [[4.897094249725342]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_9f23e5f65f2d81fcc0b69e8c24dfc3fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb42587a18ca51b292408a36234475f8
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60c8b6b6e5610ac110fcf56e564e2806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.762831211090088]], [[5.912539482116699]], [[6.511545181274414]], [[5.992820739746094]], [[6.968940258026123]], [[6.612647533416748]], [[6.161623001098633]], [[5.722076892852783]], [[6.703250408172607]], [[5.690288066864014]], [[6.0369768142700195]], [[6.201291084289551]], [[6.706302642822266]], [[5.62612771987915]], [[6.245187759399414]], [[5.9466352462768555]], [[5.768355846405029]], [[6.4490766525268555]], [[5.810674667358398]], [[5.889484405517578]], [[6.537364959716797]], [[5.689944744110107]], [[6.338402271270752]], [[7.280892372131348]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_6a4c5bfd3d6afc713c0678b8f95a670a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db333f5f66836f46eb588c1df49017d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_063c68ddc81e2e95601b9142477b2f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.857046127319336]], [[4.94091796875]], [[4.483896732330322]], [[3.889404535293579]], [[4.412776947021484]], [[3.7807023525238037]], [[3.8521409034729004]], [[4.523113250732422]], [[4.566526889801025]], [[3.31935453414917]], [[4.453447341918945]], [[4.488199234008789]], [[5.0645952224731445]], [[4.404751300811768]], [[5.043583869934082]], [[4.1073713302612305]], [[4.769421100616455]], [[4.5601487159729]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_579b5eb82256e7a422dea55c3d15d97c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.269239902496338]], [[6.215242385864258]], [[4.777440071105957]], [[5.51981258392334]], [[5.698533535003662]], [[5.038366794586182]], [[6.198485851287842]], [[5.795487403869629]], [[5.765268325805664]], [[6.254849433898926]], [[6.3785247802734375]], [[5.870089530944824]], [[5.322088241577148]], [[5.806186676025391]], [[5.142665863037109]], [[5.517502307891846]], [[6.4795145988464355]], [[5.786216735839844]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_b6a49843857d8ce0eae03c2545dcb789(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.686263084411621]], [[4.615196704864502]], [[4.329402446746826]], [[5.435118675231934]], [[4.637634754180908]], [[4.600268363952637]], [[5.243618965148926]], [[4.374950885772705]], [[4.660830974578857]], [[4.971651554107666]], [[4.599851608276367]], [[4.9320197105407715]], [[4.223817348480225]], [[4.908278465270996]], [[5.701253890991211]], [[5.603471755981445]], [[5.186424732208252]], [[5.014660358428955]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_fcca718f4c221c445f891d214cf6dbfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.9503366947174072]], [[3.9157907962799072]], [[4.34525203704834]], [[4.444269180297852]], [[4.11352014541626]], [[4.425756931304932]], [[4.358208179473877]], [[4.3368682861328125]], [[3.8690223693847656]], [[4.649980545043945]], [[4.770483493804932]], [[4.135887622833252]], [[3.9279897212982178]], [[4.586308479309082]], [[4.425254821777344]], [[5.121748924255371]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_72a6fecf1fae7a5a90f9304515d8c4b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.872812271118164]], [[4.813311576843262]], [[4.442329406738281]], [[4.180283546447754]], [[4.563350677490234]], [[4.228987693786621]], [[5.5067973136901855]], [[4.991118431091309]], [[5.469857215881348]], [[4.687140464782715]], [[5.060680389404297]], [[4.81998348236084]], [[4.570867538452148]], [[4.406635284423828]], [[5.396178722381592]], [[4.410801887512207]], [[5.008913040161133]], [[5.029808044433594]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_75465324d387cdd1640c8bf34c7bd5ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81236129c333dfe7ae73bbcbb0979cbf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8795795440673828]], [[1.038812518119812]], [[0.8776819109916687]], [[1.1465747356414795]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_74f4be8852356d7d10a119d108a98b22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.57105016708374]], [[5.187228679656982]], [[4.687733173370361]], [[4.013662338256836]], [[5.9551472663879395]], [[5.180907726287842]], [[5.969728469848633]], [[5.038147926330566]], [[5.276065349578857]], [[5.582442283630371]], [[5.576443672180176]], [[5.48445987701416]], [[5.050901889801025]], [[4.763657569885254]], [[5.620220184326172]], [[4.300808906555176]], [[5.081611633300781]], [[5.307088375091553]], [[5.391952037811279]], [[5.492855548858643]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_66af98ff53903665444eeb1746ff7a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.047842025756836]], [[3.1700327396392822]], [[3.9174647331237793]], [[3.7232398986816406]], [[3.2468199729919434]], [[4.048626899719238]], [[3.1338746547698975]], [[3.322093963623047]], [[3.064919948577881]], [[3.4265213012695312]], [[3.5901713371276855]], [[3.7378084659576416]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_39af87b860b748daf48f8027526b494f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.73850679397583]], [[4.863675594329834]], [[5.502257347106934]], [[4.718661308288574]], [[5.080923080444336]], [[4.2703328132629395]], [[5.199297904968262]], [[5.081809997558594]], [[4.6135382652282715]], [[4.608609676361084]], [[5.072863578796387]], [[4.551656246185303]], [[5.205888748168945]], [[4.730347633361816]], [[4.898443222045898]], [[5.186558246612549]], [[5.368105411529541]], [[5.535336017608643]], [[5.157254219055176]], [[5.266867637634277]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_b7033bb61b7f40638ce25a3819be9cc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6a79f19dffaf1a401b1a360fa95eb71
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.506019115447998]], [[2.3050897121429443]], [[2.5983593463897705]], [[2.21671199798584]], [[2.4672653675079346]], [[2.830336570739746]], [[2.7277143001556396]], [[2.5625483989715576]], [[2.609344959259033]], [[2.6353414058685303]], [[2.517425060272217]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_cc8e6be2d0c1640591da884ef5af2733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa5522cc0fff55438ea3c29c97b2341b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.322866916656494]], [[4.614137172698975]], [[3.6140758991241455]], [[3.754441261291504]], [[3.403918743133545]], [[3.782527446746826]], [[3.539771556854248]], [[3.5740444660186768]], [[4.168973922729492]], [[4.086240291595459]], [[4.050668239593506]], [[3.755788564682007]], [[3.9075751304626465]], [[3.614896297454834]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


    class TestPrimitiveOp_81fd2e7f23102c56c200110e8bf571f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.513790130615234]], [[4.512863636016846]], [[5.205973148345947]], [[5.517946243286133]], [[4.983819961547852]], [[6.141815185546875]], [[5.8559980392456055]], [[5.97186279296875]], [[6.138519763946533]], [[5.829843997955322]], [[5.704123497009277]], [[5.654500961303711]], [[5.876495361328125]], [[5.6354475021362305]], [[5.825729846954346]], [[5.408461093902588]], [[5.917852401733398]], [[5.67879581451416]], [[5.08447790145874]], [[5.052452087402344]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_9ffd764ac3dffdd62f6bbdfdadc35c68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34324.3359375]], [[35210.25]], [[35337.359375]], [[27557.31640625]], [[37704.00390625]], [[39159.00390625]]], [[[35315.1015625]], [[36224.921875]], [[36351.7421875]], [[28357.1328125]], [[38788.0546875]], [[40294.0859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_c77532d10bdfa68686076a1976d55c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33221.203125]], [[39309.6171875]], [[33714.2109375]], [[35410.09765625]], [[41504.390625]], [[40679.7265625]]], [[[33992.7578125]], [[40221.5]], [[34499.7421875]], [[36229.44921875]], [[42466.921875]], [[41625.83984375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_e164607806f11fd2c3192040321d6eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43299.49609375]], [[46142.76171875]], [[36165.4921875]], [[34835.8125]], [[49133.00390625]], [[27065.525390625]]], [[[44534.171875]], [[47459.6953125]], [[37196.3203125]], [[35827.3828125]], [[50539.9609375]], [[27837.369140625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_5b21d96750e54b28cc4205816885c0ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32312.3125]], [[36691.55859375]], [[38778.1171875]], [[42773.34765625]], [[37369.8203125]], [[45978.91796875]]], [[[33218.5703125]], [[37730.0859375]], [[39871.8984375]], [[43983.71875]], [[38432.6796875]], [[47276.96875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


    class TestPrimitiveOp_5f3894cc0ca4320784d39d694862cc3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.427326679229736]], [[7.228569030761719]], [[6.992318153381348]], [[6.183574199676514]], [[7.347770690917969]], [[7.364208698272705]], [[7.418874740600586]], [[7.159008026123047]], [[6.773091793060303]], [[6.469487190246582]], [[7.069914817810059]], [[7.183864593505859]], [[6.996612548828125]], [[6.526675224304199]], [[7.3109564781188965]], [[7.045126438140869]], [[7.064694404602051]], [[6.389091491699219]], [[6.853155612945557]], [[7.308903694152832]], [[6.340528964996338]], [[6.342247009277344]], [[7.179606914520264]], [[7.434704780578613]], [[6.474822521209717]], [[7.299975872039795]], [[6.992025375366211]], [[7.441059112548828]], [[7.8285698890686035]], [[7.170892715454102]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_4724080523008cb139fb79e2294a8df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.369341850280762]], [[7.440235137939453]], [[7.959526062011719]], [[6.732922554016113]], [[7.920384883880615]], [[6.782915115356445]], [[6.6313252449035645]], [[6.644609451293945]], [[8.01773738861084]], [[7.6204633712768555]], [[6.5885396003723145]], [[7.120371341705322]], [[7.276170253753662]], [[7.940833568572998]], [[7.641347885131836]], [[7.419842720031738]], [[7.180537223815918]], [[7.764443397521973]], [[6.988077163696289]], [[6.984099864959717]], [[6.495869159698486]], [[6.951446533203125]], [[7.682574272155762]], [[7.8235368728637695]], [[7.526726722717285]], [[7.461899757385254]], [[6.840702056884766]], [[7.3826093673706055]], [[7.137312889099121]], [[7.787014007568359]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_093ed3cdc8f4d1b21619a0d4aa800009(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80397bb364335531627c2a66568545dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0501ec28b4f8a34a28184cadfa448b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.031045913696289]], [[8.570387840270996]], [[7.079840660095215]], [[6.9657511711120605]], [[8.437033653259277]], [[7.539844989776611]], [[7.698031902313232]], [[7.678504943847656]], [[6.655674934387207]], [[6.759970188140869]], [[7.527650356292725]], [[7.25475549697876]], [[7.833059310913086]], [[7.854125022888184]], [[6.990649223327637]], [[7.532878875732422]], [[8.42581558227539]], [[7.625922679901123]], [[7.514565467834473]], [[8.335790634155273]], [[7.531124114990234]], [[7.539545059204102]], [[7.470608234405518]], [[7.559680461883545]], [[7.1960554122924805]], [[7.70048189163208]], [[7.439781188964844]], [[7.400554180145264]], [[8.132777214050293]], [[7.975470066070557]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_de4122b47f4d5830fb2e79fd9e7aeb1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.580674648284912]], [[7.62647008895874]], [[6.839269638061523]], [[7.1614837646484375]], [[8.016974449157715]], [[6.909478187561035]], [[7.518780708312988]], [[6.050467491149902]], [[7.179502010345459]], [[7.222623348236084]], [[7.504827976226807]], [[7.768027305603027]], [[7.591369152069092]], [[7.284604549407959]], [[7.636691570281982]], [[6.545493125915527]], [[7.5735344886779785]], [[7.230812072753906]], [[6.924068450927734]], [[7.332103252410889]], [[7.459962368011475]], [[7.344234466552734]], [[7.400725364685059]], [[6.7071123123168945]], [[7.448028087615967]], [[7.550012588500977]], [[6.99104642868042]], [[7.519863128662109]], [[8.079681396484375]], [[7.346412658691406]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_acd7fa8210f9d946973e5e79a14205e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3224000930786133]], [[2.8902981281280518]], [[2.7788562774658203]], [[3.364142894744873]], [[3.5580146312713623]], [[2.896169900894165]], [[3.005430221557617]], [[3.2257089614868164]], [[2.7983479499816895]], [[2.645906686782837]], [[3.040734052658081]], [[3.4453206062316895]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_ce4bfed7869f8fb2e72a6ff5b27b052c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.9385769367218018]], [[2.4369940757751465]], [[3.0526251792907715]], [[2.676270008087158]], [[3.005340337753296]], [[2.3457677364349365]], [[2.8266632556915283]], [[3.110970973968506]], [[2.641709566116333]], [[2.2739691734313965]], [[2.794102191925049]], [[3.12825083732605]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_a60bac86592557e95f5aa55699cc411e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.661965370178223]], [[6.609972953796387]], [[5.915094375610352]], [[6.050534725189209]], [[6.65255880355835]], [[5.39961051940918]], [[7.225243091583252]], [[6.373495101928711]], [[5.898714542388916]], [[6.293486595153809]], [[6.69158411026001]], [[5.537364482879639]], [[5.7546796798706055]], [[6.335968017578125]], [[6.212339401245117]], [[6.0673675537109375]], [[6.005062103271484]], [[5.6984543800354]], [[6.54160737991333]], [[6.147066593170166]], [[6.136723041534424]], [[6.483702182769775]], [[6.854306221008301]], [[6.514199733734131]], [[6.482630252838135]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_ad68a2677c6df4a1ddd8dd14ee134480(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.467135906219482]], [[4.3664231300354]], [[4.372544288635254]], [[4.080681324005127]], [[4.4120869636535645]], [[4.544767379760742]], [[4.381662368774414]], [[4.579371929168701]], [[4.712819576263428]], [[5.40225076675415]], [[4.953717231750488]], [[4.246953010559082]], [[5.275709629058838]], [[4.377523899078369]], [[4.259751319885254]], [[4.641990661621094]], [[4.629286766052246]], [[4.700592994689941]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_fecbb9c91fed1ac2177716e795bec60c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.9325298070907593]], [[1.4834176301956177]], [[1.8315191268920898]], [[1.4436335563659668]], [[1.6894466876983643]]]], dtype='float32').reshape([1, 5, 1, 1]),
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


    class TestPrimitiveOp_67784ff184ba081217645ef4cd419c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.3792216777801514]], [[2.585481643676758]], [[2.5880722999572754]], [[2.69108247756958]], [[2.578275442123413]], [[2.1945242881774902]], [[2.584146738052368]], [[2.9685192108154297]], [[2.8302712440490723]], [[2.489863634109497]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_6f144635b83f08dc0ec4e1dcb26b4f5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.49184513092041]], [[5.590744495391846]], [[5.003491401672363]], [[5.596516132354736]], [[5.173966407775879]], [[4.47603178024292]], [[4.616091251373291]], [[4.412466049194336]], [[4.7998127937316895]], [[4.498542308807373]], [[4.57667350769043]], [[5.419522762298584]], [[4.814596652984619]], [[4.749876499176025]], [[5.119980812072754]], [[5.1041579246521]], [[4.085983753204346]], [[5.195230484008789]], [[4.809571266174316]], [[4.356119632720947]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_cda5ac54979b1ee820aea721ca022ddf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.798402309417725]], [[6.471826076507568]], [[6.2061543464660645]], [[6.678011417388916]], [[6.453592777252197]], [[6.86955451965332]], [[6.800777912139893]], [[6.89926290512085]], [[7.662395000457764]], [[6.002626419067383]], [[6.689345359802246]], [[6.529068946838379]], [[6.422098159790039]], [[6.518607139587402]], [[6.604228973388672]], [[6.671124458312988]], [[6.413301467895508]], [[6.887411594390869]], [[7.293798446655273]], [[7.487094402313232]], [[6.487042427062988]], [[6.869065761566162]], [[6.210934162139893]], [[6.379401206970215]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_761e0693c7faa426ce89a24460e11a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08b32bc10ed16b6e1064cab002e01fc8
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00367c08d8e30c08d1d03ce7d8b6ffd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5114879608154297]], [[3.2385427951812744]], [[3.022507905960083]], [[2.5118393898010254]], [[3.1523807048797607]], [[2.381998300552368]], [[3.2308504581451416]], [[3.1900980472564697]], [[2.7136454582214355]], [[3.152656078338623]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_29ace9b287f423c2e381517f90cd8008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0be4fdd9128c2c1a4eadb94c63a30ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.349632740020752]], [[4.581267356872559]], [[4.542279243469238]], [[4.27977180480957]], [[4.8218674659729]], [[3.832540512084961]], [[4.169091701507568]], [[4.272579669952393]], [[4.45680570602417]], [[4.216972351074219]], [[4.61435079574585]], [[4.593276023864746]], [[3.9209370613098145]], [[4.463364601135254]], [[4.558143138885498]], [[3.674036979675293]], [[4.767111778259277]], [[3.4685769081115723]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_cc80fd33c3cba3452357c4ea18c51e24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_831862bc6d25ea81d4a94454e1a13a3c
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.318058967590332, 7.742279052734375, 7.49235200881958, 7.212325096130371, 7.49655294418335, 7.519100189208984, 7.2283406257629395, 7.083418369293213, 6.941357135772705, 6.601337909698486, 6.675265789031982, 7.745193004608154, 7.000194549560547, 7.5520853996276855, 7.6470723152160645, 6.704595565795898, 7.123214244842529, 8.127568244934082, 7.244419574737549, 8.030851364135742, 7.505814552307129, 7.365539073944092, 9.035940170288086, 7.207608222961426, 7.554633617401123, 6.867268085479736, 8.027496337890625, 7.540754318237305, 7.7998857498168945, 7.971572399139404]], dtype='float32').reshape([1, 30]),
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


    class TestPrimitiveOp_84ced437b957227b6b126e140732fc52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.729344367980957]], [[8.053105354309082]], [[7.784078598022461]], [[7.947120666503906]], [[8.634868621826172]], [[8.28966236114502]], [[7.792207717895508]], [[8.541390419006348]], [[8.528681755065918]], [[7.953497409820557]], [[8.183775901794434]], [[7.91998815536499]], [[7.978310585021973]], [[7.298711776733398]], [[8.158519744873047]], [[7.805086612701416]], [[7.681483268737793]], [[7.892937660217285]], [[8.441560745239258]], [[7.774263858795166]], [[8.654806137084961]], [[8.259748458862305]], [[7.420984745025635]], [[8.28664493560791]], [[8.260310173034668]], [[7.600845813751221]], [[8.31584644317627]], [[7.932604789733887]], [[6.945162773132324]], [[7.285499572753906]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_fd7a0217b4cc3867f1338a7dcbcab42f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa6a208551763b029a4175fcd015eae
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4133658409118652]], [[1.4126325845718384]], [[1.956082820892334]], [[1.3447970151901245]], [[1.6026768684387207]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_0ee8530bdcb0e57213d36b14c8112b22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb78498383eaa8c94e61c1589cccd4d3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6246941089630127]], [[2.544447898864746]], [[2.795630693435669]], [[2.9004950523376465]], [[2.514697790145874]], [[2.9879279136657715]], [[2.600862503051758]], [[2.8251161575317383]], [[2.3790998458862305]], [[2.588230609893799]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_a03ac9a24acd2ae2998ee59256bd56c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.971728324890137]], [[5.582655429840088]], [[5.47714376449585]], [[6.247722625732422]], [[5.601855754852295]], [[5.2803473472595215]], [[4.945881366729736]], [[6.030325889587402]], [[5.071672439575195]], [[5.13779354095459]], [[5.508490562438965]], [[5.381728172302246]], [[5.397087574005127]], [[4.520235061645508]], [[4.920662879943848]], [[5.529647350311279]], [[5.867205619812012]], [[5.492212295532227]], [[5.60070276260376]], [[4.916013240814209]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_355d540e924cca0e725b3eff63023920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda5a952e5801a27bc5b8a72b8de5ce0
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e62785514a5af7bcd8ca61a6cef0ee6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a124f1c5540890bc8b3742770aa7f68
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.0072021484375]], [[4.442929744720459]], [[4.372024059295654]], [[3.8059115409851074]], [[3.8065803050994873]], [[3.36421537399292]], [[4.133330821990967]], [[3.93577241897583]], [[4.406898498535156]], [[4.474634170532227]], [[4.070117950439453]], [[4.775428295135498]], [[4.3750386238098145]], [[3.9665915966033936]], [[3.853419065475464]], [[4.157354354858398]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_d52950df7d24ecccc3a9d58003d58cb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa5522cc0fff55438ea3c29c97b2341b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.294074535369873]], [[4.268002033233643]], [[3.913578987121582]], [[4.219417572021484]], [[4.341715335845947]], [[4.475269794464111]], [[4.291004657745361]], [[4.3809051513671875]], [[3.1958839893341064]], [[3.4503233432769775]], [[4.014822959899902]], [[3.992638111114502]], [[3.7971768379211426]], [[3.768758535385132]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_856e7db8079e21db5650e9e9ad9f9c94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f627817c4eadec5a41ef9ecacf7e37bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.350203514099121]], [[4.70102071762085]], [[4.559035778045654]], [[4.141345024108887]], [[4.519817352294922]], [[4.664248466491699]], [[4.759017467498779]], [[3.98451828956604]], [[4.821098804473877]], [[4.811173915863037]], [[5.116090297698975]], [[4.81430721282959]], [[3.7403039932250977]], [[4.816918849945068]], [[4.844472885131836]], [[5.181200981140137]], [[4.626385688781738]], [[4.89996337890625]], [[4.556179523468018]], [[4.963408946990967]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_add7cf5d857a91e0208b6d35c351493a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1df71eff0cfaeb78bbec10af6b5c3060
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.00551986694336]], [[8.049155235290527]], [[7.384499549865723]], [[8.339685440063477]], [[7.984127998352051]], [[8.674389839172363]], [[8.147443771362305]], [[7.1571455001831055]], [[8.28207778930664]], [[7.843662261962891]], [[7.487427711486816]], [[7.830111503601074]], [[7.376617431640625]], [[8.312586784362793]], [[8.032049179077148]], [[8.098679542541504]], [[8.163570404052734]], [[7.175169467926025]], [[7.265573978424072]], [[7.794568061828613]], [[7.623687744140625]], [[7.449440002441406]], [[7.246952056884766]], [[7.624670028686523]], [[7.131568431854248]], [[7.504866123199463]], [[8.205310821533203]], [[8.012397766113281]], [[8.011343002319336]], [[7.762167930603027]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_bf7f3296492e24dc4146271790f4883e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.434598922729492]], [[6.018231391906738]], [[6.290614128112793]], [[6.7121357917785645]], [[5.903967380523682]], [[5.9446611404418945]], [[5.652713298797607]], [[6.595141887664795]], [[6.70281982421875]], [[6.8267822265625]], [[7.258833885192871]], [[5.433291912078857]], [[6.281140327453613]], [[6.249874591827393]], [[5.903461456298828]], [[6.033217430114746]], [[6.7142720222473145]], [[7.203909397125244]], [[6.759840965270996]], [[6.176536560058594]], [[5.412111759185791]], [[6.212423324584961]], [[6.020208835601807]], [[6.183920860290527]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_c66c68970f1c93cbd8bb5e2bdf391b73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1633780f35761e3522fa8ab2a1b4e37
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.44723653793335]], [[6.102865695953369]], [[5.985477447509766]], [[6.842640399932861]], [[5.418141841888428]], [[5.740512371063232]], [[5.729429721832275]], [[6.500969886779785]], [[6.418935775756836]], [[5.616406440734863]], [[5.975831031799316]], [[5.51600456237793]], [[6.463415622711182]], [[6.480999946594238]], [[5.885873317718506]], [[5.866087436676025]], [[5.8708038330078125]], [[5.90671443939209]], [[7.097609996795654]], [[5.892422199249268]], [[6.3885416984558105]], [[5.939206123352051]], [[5.946408271789551]], [[5.895269393920898]], [[6.106153964996338]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_eaf5b258cdf7d6c2e016b1114ad27757(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dd532fa575b929ebb34e47477432042
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5717320442199707]], [[3.431580066680908]], [[2.5586700439453125]], [[3.248905897140503]], [[2.9694464206695557]], [[3.5539824962615967]], [[3.0985803604125977]], [[2.924131393432617]], [[3.2537245750427246]], [[2.9115447998046875]], [[2.956376552581787]], [[3.03542423248291]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


    class TestPrimitiveOp_db172a40cd495a3cb09c705a511cdd98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[688.7190551757812]], [[785.6703491210938]], [[735.470703125]], [[731.02197265625]], [[765.8663330078125]], [[660.8710327148438]], [[704.5062255859375]], [[700.3460693359375]], [[710.5972290039062]], [[776.5182495117188]], [[802.5549926757812]], [[751.0782470703125]], [[713.1818237304688]], [[680.6382446289062]], [[728.703857421875]], [[665.543212890625]], [[712.4335327148438]], [[682.6065063476562]], [[744.167724609375]], [[741.5267944335938]], [[755.0457763671875]], [[734.9337768554688]], [[714.0075073242188]], [[708.40283203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_8dc4d19a2e585354f26950106c7d1c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[72.65186309814453]], [[91.7384262084961]], [[89.04048156738281]], [[86.67623138427734]], [[81.67875671386719]], [[85.5860824584961]], [[85.38785552978516]], [[87.42566680908203]], [[86.80927276611328]], [[94.9423599243164]], [[92.74808502197266]], [[80.63028717041016]], [[84.66590881347656]], [[92.82050323486328]], [[92.692138671875]], [[80.77722930908203]], [[89.87425231933594]], [[80.3315658569336]], [[93.5545654296875]], [[93.15975952148438]], [[82.20037841796875]], [[89.98966217041016]], [[84.42134857177734]], [[85.27412414550781]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_05ed4f7322f16e66ff4bce23d03c9b67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[44.67763900756836]], [[44.78001022338867]], [[48.8891716003418]], [[47.86702346801758]], [[50.295352935791016]], [[46.41169357299805]], [[40.6933708190918]], [[45.33034133911133]], [[46.9334602355957]], [[44.16217803955078]], [[38.59510040283203]], [[49.060630798339844]], [[43.3294563293457]], [[47.460296630859375]], [[41.22036361694336]], [[42.97414016723633]], [[50.949039459228516]], [[42.96493148803711]], [[43.61009979248047]], [[46.273555755615234]], [[46.58684539794922]], [[47.14780044555664]], [[47.21940231323242]], [[48.132850646972656]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f2f7ba650c292f21b69f592c4dbfdb6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[22.74772834777832]], [[23.626388549804688]], [[22.602088928222656]], [[21.93589973449707]], [[21.32398796081543]], [[22.9318904876709]], [[21.411766052246094]], [[21.45660972595215]], [[23.392732620239258]], [[23.401206970214844]], [[23.38231086730957]], [[22.44701385498047]], [[23.534521102905273]], [[21.442602157592773]], [[19.88401985168457]], [[22.459274291992188]], [[22.3023681640625]], [[21.33476448059082]], [[19.748985290527344]], [[21.4251708984375]], [[21.819591522216797]], [[22.805885314941406]], [[22.18486213684082]], [[22.38351821899414]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_310cc395dee562be1c63997ad8024069(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[28815.861328125]], [[30566.94140625]], [[30025.67578125]], [[31712.501953125]], [[38763.7109375]], [[29146.6328125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_6500797e24605dd09bfb02b5489b5b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32805.8515625]], [[37008.89453125]], [[41284.26171875]], [[31982.39453125]], [[35843.10546875]], [[39359.27734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_7e07cf355c72eaa83f0b3089e4277e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[38334.63671875]], [[42924.8359375]], [[41756.01953125]], [[38537.6953125]], [[44312.58984375]], [[51708.421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_5794873b6dda3d23d87378a53a655d86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d06a79e0a9078d999d99f26769322018
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41016.546875]], [[40018.828125]], [[41046.40625]], [[50225.1796875]], [[46902.21484375]], [[44018.38671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


    class TestPrimitiveOp_ce5e646adbd80cba471908dfb7fed0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d8a5e6d8621729e88ac2bc66e0bf204
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.235050678253174]], [[6.788114547729492]], [[6.863269805908203]], [[6.1629228591918945]], [[6.510852813720703]], [[6.815029144287109]], [[6.328566551208496]], [[5.321353435516357]], [[6.496885776519775]], [[6.512045860290527]], [[5.656451225280762]], [[6.609005451202393]], [[6.846685409545898]], [[6.689188480377197]], [[6.578765869140625]], [[6.044051170349121]], [[6.097584247589111]], [[6.787467956542969]], [[6.206361770629883]], [[6.546125411987305]], [[6.848434925079346]], [[6.219845771789551]], [[6.587372303009033]], [[6.3889946937561035]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_6b9191676786c97235f0eaec6ec5895b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa5482916cffc526445466b37baf168
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.516043186187744, 4.707422256469727, 5.164888858795166, 5.325064182281494, 5.265877723693848, 5.291617393493652, 5.1853132247924805, 5.42218017578125, 4.414474010467529, 4.513939380645752, 5.1259446144104, 4.9605937004089355, 4.702442169189453, 5.391978740692139, 5.0975661277771, 5.194991588592529, 5.173627853393555, 4.850863933563232]], dtype='float32').reshape([1, 18]),
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


    class TestPrimitiveOp_82c720832f56053f6a8e6cbb2fefab2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e722c534a85f6265bab20f6209cc641
        def get_inputs(self):
            return [
                paddle.to_tensor([[6.707153797149658, 6.260829448699951, 6.551505088806152, 6.331302642822266, 6.301877975463867, 6.822994709014893, 6.100552558898926, 6.320962905883789, 5.698636531829834, 6.4518914222717285, 5.776687145233154, 7.555084705352783, 5.768364906311035, 6.92980432510376, 6.29964542388916, 5.982891082763672, 6.051077365875244, 6.072269916534424, 5.807013034820557, 6.226430416107178, 6.443246364593506, 6.073713302612305, 6.26926326751709]], dtype='float32').reshape([1, 23]),
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


    class TestPrimitiveOp_8d51a76f735611d1e5cbc5f5ed988f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.477514266967773]], [[7.119456768035889]], [[7.085227012634277]], [[6.6396708488464355]], [[7.529614448547363]], [[7.339506149291992]], [[7.696058750152588]], [[7.32058572769165]], [[7.314259052276611]], [[7.595938205718994]], [[7.538951873779297]], [[6.933803081512451]], [[7.610673427581787]], [[7.8937482833862305]], [[7.015742301940918]], [[8.132213592529297]], [[7.123446941375732]], [[7.175824165344238]], [[7.174844741821289]], [[7.820660591125488]], [[7.858874797821045]], [[6.548339366912842]], [[7.482433319091797]], [[7.067334175109863]], [[7.9104390144348145]], [[6.727339267730713]], [[6.801524639129639]], [[7.090723037719727]], [[7.15854024887085]], [[7.843683242797852]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_ff9733ef99bc43c0ad322fad84cbdf14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.417559623718262]], [[7.743318557739258]], [[7.118884086608887]], [[7.612246036529541]], [[7.387399196624756]], [[8.145689010620117]], [[8.002328872680664]], [[7.574892520904541]], [[7.7750959396362305]], [[7.651700019836426]], [[7.327408790588379]], [[6.881934642791748]], [[8.146565437316895]], [[6.961785793304443]], [[7.403148651123047]], [[7.219143390655518]], [[7.513122081756592]], [[7.609578609466553]], [[8.403047561645508]], [[8.094966888427734]], [[7.281956672668457]], [[6.785111427307129]], [[7.188939094543457]], [[7.072811603546143]], [[7.33947229385376]], [[7.32002592086792]], [[7.809920787811279]], [[7.630326747894287]], [[7.602578163146973]], [[7.1075544357299805]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_67632bbc28fac08d87e2b01bbd94d441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.369239330291748]], [[1.2205345630645752]], [[1.3513569831848145]], [[1.3422176837921143]], [[1.3516736030578613]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_0812edbead67fd3e54d3016ecc68d069(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.545330047607422]], [[2.542398691177368]], [[2.236086130142212]], [[2.722609519958496]], [[2.6072356700897217]], [[3.208588123321533]], [[3.3306355476379395]], [[2.2302825450897217]], [[2.476623058319092]], [[2.767191171646118]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_610b111e0e3424fa288d4add6b6e79a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.671944618225098]], [[6.098160743713379]], [[5.53664493560791]], [[5.733607292175293]], [[6.2740702629089355]], [[5.394620895385742]], [[5.611918926239014]], [[5.351341724395752]], [[6.054460525512695]], [[5.389400959014893]], [[5.4167094230651855]], [[5.1283745765686035]], [[5.846133232116699]], [[5.851635456085205]], [[5.39577579498291]], [[6.457047462463379]], [[6.039535045623779]], [[6.229255676269531]], [[5.209201335906982]], [[5.494519233703613]], [[6.435673236846924]], [[6.3030104637146]], [[4.4522929191589355]], [[6.078214645385742]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_1cbd0c5139cffd83614d4ed5aa51ba0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.120477676391602]], [[4.59541130065918]], [[4.452274799346924]], [[3.8959805965423584]], [[4.421902179718018]], [[4.485260963439941]], [[5.113558769226074]], [[4.413151264190674]], [[4.051578521728516]], [[4.178587436676025]], [[4.9391093254089355]], [[4.186689853668213]], [[4.489201068878174]], [[4.360198497772217]], [[4.403806686401367]], [[4.735654354095459]], [[4.867974281311035]], [[4.216265678405762]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_60f1923d78cfd081c20aa4ac9471899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae332de3c33d5ad1aaa05f2733f02416
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca3cb228da68f1be69472813619ee572(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.3108625411987305]], [[6.79969596862793]], [[6.751509666442871]], [[6.1104865074157715]], [[6.556639194488525]], [[6.563843250274658]], [[6.04982328414917]], [[6.986202716827393]], [[6.39040470123291]], [[6.591248035430908]], [[6.4414215087890625]], [[6.0124945640563965]], [[6.929512977600098]], [[6.34401798248291]], [[7.0286431312561035]], [[6.831506252288818]], [[5.958920955657959]], [[5.804043292999268]], [[5.180810928344727]], [[5.40272331237793]], [[6.816746711730957]], [[5.976834774017334]], [[6.840426445007324]], [[6.716890811920166]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_110dab3be8671deea3a2aaa8e0c73d0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9a5be699233ae66d2644ef9ba39603
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8333555459976196]], [[0.4858071208000183]], [[1.1957030296325684]], [[1.2955039739608765]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_fec96e81068eb7c327d8a224b2e09062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85b01dc5b87514b414a7a8aee3b35d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5765128135681152]], [[2.6508169174194336]], [[2.5669305324554443]], [[2.265747547149658]], [[2.859575033187866]], [[3.018110752105713]], [[2.4999642372131348]], [[2.6135239601135254]], [[2.8138022422790527]], [[2.400343894958496]], [[2.293578863143921]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_080eee1221cb1e986576c07df70c56a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.678655624389648]], [[7.421043872833252]], [[7.546217441558838]], [[7.017660140991211]], [[7.533844947814941]], [[7.912869453430176]], [[7.165874004364014]], [[7.757705211639404]], [[7.679967880249023]], [[7.868678569793701]], [[7.383896350860596]], [[7.19304895401001]], [[7.269775390625]], [[7.78610897064209]], [[7.44060754776001]], [[7.792629241943359]], [[6.972398281097412]], [[7.668750286102295]], [[7.87344217300415]], [[7.160478591918945]], [[7.612797737121582]], [[7.538360118865967]], [[7.100478172302246]], [[7.216472148895264]], [[7.498114109039307]], [[7.365005970001221]], [[7.5870442390441895]], [[7.954311847686768]], [[6.997818470001221]], [[6.68562650680542]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_3f4a873facb1047c34eaa868eb5963e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.204582214355469]], [[4.17450475692749]], [[4.450677394866943]], [[5.001364707946777]], [[4.378679275512695]], [[4.139589786529541]], [[4.008517742156982]], [[4.55598258972168]], [[4.8348588943481445]], [[3.643404006958008]], [[5.065579891204834]], [[4.885112285614014]], [[4.120347499847412]], [[4.529646873474121]], [[3.988499879837036]], [[3.9207963943481445]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_c1dd30f4990e1fad2afb890e3f587b76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.332149505615234]], [[8.124646186828613]], [[9.008127212524414]], [[7.852988243103027]], [[7.852867126464844]], [[7.848668575286865]], [[8.38390064239502]], [[7.666966915130615]], [[8.083961486816406]], [[8.015180587768555]], [[8.18043327331543]], [[7.053821086883545]], [[7.988963603973389]], [[7.258120536804199]], [[7.686677932739258]], [[7.7280659675598145]], [[7.685739040374756]], [[7.41143274307251]], [[6.809802532196045]], [[8.155046463012695]], [[6.916194438934326]], [[7.524030685424805]], [[8.538736343383789]], [[7.1718244552612305]], [[7.731924057006836]], [[7.71543550491333]], [[7.578956604003906]], [[7.180920124053955]], [[7.5848236083984375]], [[7.148171424865723]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_6b26cb015622fc55958392c6bd6c5d7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.464395046234131]], [[6.970254421234131]], [[6.331833839416504]], [[6.208786487579346]], [[6.412256240844727]], [[6.44594144821167]], [[6.668659687042236]], [[5.506393909454346]], [[5.857957363128662]], [[6.159074783325195]], [[6.215778350830078]], [[7.449837684631348]], [[6.134645462036133]], [[7.1130499839782715]], [[6.447407245635986]], [[6.364443778991699]], [[6.062101364135742]], [[6.829754829406738]], [[6.665282249450684]], [[5.587331295013428]], [[6.970404624938965]], [[6.038416385650635]], [[6.559352397918701]], [[7.202384948730469]], [[6.573240756988525]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_c7da8b5ff1a11e5e6aecec73df44fc79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.648985385894775]], [[4.977217197418213]], [[4.969912052154541]], [[4.990406513214111]], [[5.3486456871032715]], [[5.269417762756348]], [[5.3413591384887695]], [[4.685579776763916]], [[4.341116905212402]], [[5.372086524963379]], [[4.675849914550781]], [[5.339875221252441]], [[4.787026405334473]], [[4.957879066467285]], [[4.992903709411621]], [[4.947808742523193]], [[4.469698905944824]], [[3.9869487285614014]], [[4.38286018371582]], [[5.121321201324463]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_1e6dc5f904aebe631910c1b55c6a5abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.745029449462891]], [[4.362461090087891]], [[4.826979160308838]], [[4.8406877517700195]], [[4.93057918548584]], [[4.475962162017822]], [[4.890925407409668]], [[4.360519886016846]], [[5.017579555511475]], [[5.275971412658691]], [[4.956449508666992]], [[4.257177829742432]], [[4.3468146324157715]], [[4.757590293884277]], [[4.607741355895996]], [[4.7346930503845215]], [[5.006930828094482]], [[5.1289238929748535]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_a99f73dfc5916b865f07012e0d48e524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.018347263336182]], [[5.169200897216797]], [[4.998176097869873]], [[4.5034708976745605]], [[4.442930221557617]], [[5.493774890899658]], [[5.000889301300049]], [[4.583656311035156]], [[5.095301628112793]], [[5.118431568145752]], [[4.825158596038818]], [[4.5962653160095215]], [[4.8283796310424805]], [[4.349386215209961]], [[4.9702229499816895]], [[4.948547840118408]], [[4.366069793701172]], [[4.897094249725342]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_177aab49fa33a6f90da440e5529d03ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc8405c8dc00d11843d8deda02d87197
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a292d7387cd3aed47ddcff3fa2b94100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.762831211090088]], [[5.912539482116699]], [[6.511545181274414]], [[5.992820739746094]], [[6.968940258026123]], [[6.612647533416748]], [[6.161623001098633]], [[5.722076892852783]], [[6.703250408172607]], [[5.690288066864014]], [[6.0369768142700195]], [[6.201291084289551]], [[6.706302642822266]], [[5.62612771987915]], [[6.245187759399414]], [[5.9466352462768555]], [[5.768355846405029]], [[6.4490766525268555]], [[5.810674667358398]], [[5.889484405517578]], [[6.537364959716797]], [[5.689944744110107]], [[6.338402271270752]], [[7.280892372131348]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_958361eff9c4bbc0da757b7a8cfc275e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.857046127319336]], [[4.94091796875]], [[4.483896732330322]], [[3.889404535293579]], [[4.412776947021484]], [[3.7807023525238037]], [[3.8521409034729004]], [[4.523113250732422]], [[4.566526889801025]], [[3.31935453414917]], [[4.453447341918945]], [[4.488199234008789]], [[5.0645952224731445]], [[4.404751300811768]], [[5.043583869934082]], [[4.1073713302612305]], [[4.769421100616455]], [[4.5601487159729]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_6550db09783c48659f4da1d503a365bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.269239902496338]], [[6.215242385864258]], [[4.777440071105957]], [[5.51981258392334]], [[5.698533535003662]], [[5.038366794586182]], [[6.198485851287842]], [[5.795487403869629]], [[5.765268325805664]], [[6.254849433898926]], [[6.3785247802734375]], [[5.870089530944824]], [[5.322088241577148]], [[5.806186676025391]], [[5.142665863037109]], [[5.517502307891846]], [[6.4795145988464355]], [[5.786216735839844]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_970a101dc1e87c0eaf076cc5d977cf11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.686263084411621]], [[4.615196704864502]], [[4.329402446746826]], [[5.435118675231934]], [[4.637634754180908]], [[4.600268363952637]], [[5.243618965148926]], [[4.374950885772705]], [[4.660830974578857]], [[4.971651554107666]], [[4.599851608276367]], [[4.9320197105407715]], [[4.223817348480225]], [[4.908278465270996]], [[5.701253890991211]], [[5.603471755981445]], [[5.186424732208252]], [[5.014660358428955]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_c1ad4d36f0c68bbfc752ecdd93a435bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.9503366947174072]], [[3.9157907962799072]], [[4.34525203704834]], [[4.444269180297852]], [[4.11352014541626]], [[4.425756931304932]], [[4.358208179473877]], [[4.3368682861328125]], [[3.8690223693847656]], [[4.649980545043945]], [[4.770483493804932]], [[4.135887622833252]], [[3.9279897212982178]], [[4.586308479309082]], [[4.425254821777344]], [[5.121748924255371]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_55c8ffb5c412e0e7685fb9df81daaf25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.872812271118164]], [[4.813311576843262]], [[4.442329406738281]], [[4.180283546447754]], [[4.563350677490234]], [[4.228987693786621]], [[5.5067973136901855]], [[4.991118431091309]], [[5.469857215881348]], [[4.687140464782715]], [[5.060680389404297]], [[4.81998348236084]], [[4.570867538452148]], [[4.406635284423828]], [[5.396178722381592]], [[4.410801887512207]], [[5.008913040161133]], [[5.029808044433594]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_e3cf463cb9e607fd3ec561242add645f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d9a5be699233ae66d2644ef9ba39603
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8795795440673828]], [[1.038812518119812]], [[0.8776819109916687]], [[1.1465747356414795]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_73c80fa1f0bc655cb7840810d6f6d0a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.57105016708374]], [[5.187228679656982]], [[4.687733173370361]], [[4.013662338256836]], [[5.9551472663879395]], [[5.180907726287842]], [[5.969728469848633]], [[5.038147926330566]], [[5.276065349578857]], [[5.582442283630371]], [[5.576443672180176]], [[5.48445987701416]], [[5.050901889801025]], [[4.763657569885254]], [[5.620220184326172]], [[4.300808906555176]], [[5.081611633300781]], [[5.307088375091553]], [[5.391952037811279]], [[5.492855548858643]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_e491a961176f42b98cbe5912eac15837(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.047842025756836]], [[3.1700327396392822]], [[3.9174647331237793]], [[3.7232398986816406]], [[3.2468199729919434]], [[4.048626899719238]], [[3.1338746547698975]], [[3.322093963623047]], [[3.064919948577881]], [[3.4265213012695312]], [[3.5901713371276855]], [[3.7378084659576416]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_7b7b690e00425015f5b8282397b593ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.73850679397583]], [[4.863675594329834]], [[5.502257347106934]], [[4.718661308288574]], [[5.080923080444336]], [[4.2703328132629395]], [[5.199297904968262]], [[5.081809997558594]], [[4.6135382652282715]], [[4.608609676361084]], [[5.072863578796387]], [[4.551656246185303]], [[5.205888748168945]], [[4.730347633361816]], [[4.898443222045898]], [[5.186558246612549]], [[5.368105411529541]], [[5.535336017608643]], [[5.157254219055176]], [[5.266867637634277]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_2e581fbcfdd9edd7ef80ffeda658ba52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85b01dc5b87514b414a7a8aee3b35d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.506019115447998]], [[2.3050897121429443]], [[2.5983593463897705]], [[2.21671199798584]], [[2.4672653675079346]], [[2.830336570739746]], [[2.7277143001556396]], [[2.5625483989715576]], [[2.609344959259033]], [[2.6353414058685303]], [[2.517425060272217]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_3dcf53a7de371ea79d4723b44c92688b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96dc8643dc29e249b7d4dda0732345c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.322866916656494]], [[4.614137172698975]], [[3.6140758991241455]], [[3.754441261291504]], [[3.403918743133545]], [[3.782527446746826]], [[3.539771556854248]], [[3.5740444660186768]], [[4.168973922729492]], [[4.086240291595459]], [[4.050668239593506]], [[3.755788564682007]], [[3.9075751304626465]], [[3.614896297454834]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


    class TestPrimitiveOp_6febb677cff5d6c707d09e1443ae06b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.513790130615234]], [[4.512863636016846]], [[5.205973148345947]], [[5.517946243286133]], [[4.983819961547852]], [[6.141815185546875]], [[5.8559980392456055]], [[5.97186279296875]], [[6.138519763946533]], [[5.829843997955322]], [[5.704123497009277]], [[5.654500961303711]], [[5.876495361328125]], [[5.6354475021362305]], [[5.825729846954346]], [[5.408461093902588]], [[5.917852401733398]], [[5.67879581451416]], [[5.08447790145874]], [[5.052452087402344]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_614627ce4db38aa63a3263b3c850e585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34324.3359375]], [[35210.25]], [[35337.359375]], [[27557.31640625]], [[37704.00390625]], [[39159.00390625]]], [[[35315.1015625]], [[36224.921875]], [[36351.7421875]], [[28357.1328125]], [[38788.0546875]], [[40294.0859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_1e3040fd3e1544129c431d4654b862ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33221.203125]], [[39309.6171875]], [[33714.2109375]], [[35410.09765625]], [[41504.390625]], [[40679.7265625]]], [[[33992.7578125]], [[40221.5]], [[34499.7421875]], [[36229.44921875]], [[42466.921875]], [[41625.83984375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_6c4773fc36a704c064940246bb3e9f0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43299.49609375]], [[46142.76171875]], [[36165.4921875]], [[34835.8125]], [[49133.00390625]], [[27065.525390625]]], [[[44534.171875]], [[47459.6953125]], [[37196.3203125]], [[35827.3828125]], [[50539.9609375]], [[27837.369140625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_ef82a5010d88736b2a98b1f15547c2e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56177490843e69977abb19362dd06d6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32312.3125]], [[36691.55859375]], [[38778.1171875]], [[42773.34765625]], [[37369.8203125]], [[45978.91796875]]], [[[33218.5703125]], [[37730.0859375]], [[39871.8984375]], [[43983.71875]], [[38432.6796875]], [[47276.96875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


    class TestPrimitiveOp_142f01ee5400fc068441ed4b1ea0ac71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.427326679229736]], [[7.228569030761719]], [[6.992318153381348]], [[6.183574199676514]], [[7.347770690917969]], [[7.364208698272705]], [[7.418874740600586]], [[7.159008026123047]], [[6.773091793060303]], [[6.469487190246582]], [[7.069914817810059]], [[7.183864593505859]], [[6.996612548828125]], [[6.526675224304199]], [[7.3109564781188965]], [[7.045126438140869]], [[7.064694404602051]], [[6.389091491699219]], [[6.853155612945557]], [[7.308903694152832]], [[6.340528964996338]], [[6.342247009277344]], [[7.179606914520264]], [[7.434704780578613]], [[6.474822521209717]], [[7.299975872039795]], [[6.992025375366211]], [[7.441059112548828]], [[7.8285698890686035]], [[7.170892715454102]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_a229f19ff2e5a0cf7d40f588377d66d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.369341850280762]], [[7.440235137939453]], [[7.959526062011719]], [[6.732922554016113]], [[7.920384883880615]], [[6.782915115356445]], [[6.6313252449035645]], [[6.644609451293945]], [[8.01773738861084]], [[7.6204633712768555]], [[6.5885396003723145]], [[7.120371341705322]], [[7.276170253753662]], [[7.940833568572998]], [[7.641347885131836]], [[7.419842720031738]], [[7.180537223815918]], [[7.764443397521973]], [[6.988077163696289]], [[6.984099864959717]], [[6.495869159698486]], [[6.951446533203125]], [[7.682574272155762]], [[7.8235368728637695]], [[7.526726722717285]], [[7.461899757385254]], [[6.840702056884766]], [[7.3826093673706055]], [[7.137312889099121]], [[7.787014007568359]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_ed167438a4a6fb13b753203bd806c813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.031045913696289]], [[8.570387840270996]], [[7.079840660095215]], [[6.9657511711120605]], [[8.437033653259277]], [[7.539844989776611]], [[7.698031902313232]], [[7.678504943847656]], [[6.655674934387207]], [[6.759970188140869]], [[7.527650356292725]], [[7.25475549697876]], [[7.833059310913086]], [[7.854125022888184]], [[6.990649223327637]], [[7.532878875732422]], [[8.42581558227539]], [[7.625922679901123]], [[7.514565467834473]], [[8.335790634155273]], [[7.531124114990234]], [[7.539545059204102]], [[7.470608234405518]], [[7.559680461883545]], [[7.1960554122924805]], [[7.70048189163208]], [[7.439781188964844]], [[7.400554180145264]], [[8.132777214050293]], [[7.975470066070557]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_67527d68ef3c5515c4dea63e28e82b21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.580674648284912]], [[7.62647008895874]], [[6.839269638061523]], [[7.1614837646484375]], [[8.016974449157715]], [[6.909478187561035]], [[7.518780708312988]], [[6.050467491149902]], [[7.179502010345459]], [[7.222623348236084]], [[7.504827976226807]], [[7.768027305603027]], [[7.591369152069092]], [[7.284604549407959]], [[7.636691570281982]], [[6.545493125915527]], [[7.5735344886779785]], [[7.230812072753906]], [[6.924068450927734]], [[7.332103252410889]], [[7.459962368011475]], [[7.344234466552734]], [[7.400725364685059]], [[6.7071123123168945]], [[7.448028087615967]], [[7.550012588500977]], [[6.99104642868042]], [[7.519863128662109]], [[8.079681396484375]], [[7.346412658691406]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_dfa8d7ebf4af764a36eb10ebc094d4b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3224000930786133]], [[2.8902981281280518]], [[2.7788562774658203]], [[3.364142894744873]], [[3.5580146312713623]], [[2.896169900894165]], [[3.005430221557617]], [[3.2257089614868164]], [[2.7983479499816895]], [[2.645906686782837]], [[3.040734052658081]], [[3.4453206062316895]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_3678808416ec897d598599540017b710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.9385769367218018]], [[2.4369940757751465]], [[3.0526251792907715]], [[2.676270008087158]], [[3.005340337753296]], [[2.3457677364349365]], [[2.8266632556915283]], [[3.110970973968506]], [[2.641709566116333]], [[2.2739691734313965]], [[2.794102191925049]], [[3.12825083732605]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_c1bbbc2e8dd6beec3666caf2c6ac704a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.661965370178223]], [[6.609972953796387]], [[5.915094375610352]], [[6.050534725189209]], [[6.65255880355835]], [[5.39961051940918]], [[7.225243091583252]], [[6.373495101928711]], [[5.898714542388916]], [[6.293486595153809]], [[6.69158411026001]], [[5.537364482879639]], [[5.7546796798706055]], [[6.335968017578125]], [[6.212339401245117]], [[6.0673675537109375]], [[6.005062103271484]], [[5.6984543800354]], [[6.54160737991333]], [[6.147066593170166]], [[6.136723041534424]], [[6.483702182769775]], [[6.854306221008301]], [[6.514199733734131]], [[6.482630252838135]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_e56b12e5d8e2dde84035ae70da96c588(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.467135906219482]], [[4.3664231300354]], [[4.372544288635254]], [[4.080681324005127]], [[4.4120869636535645]], [[4.544767379760742]], [[4.381662368774414]], [[4.579371929168701]], [[4.712819576263428]], [[5.40225076675415]], [[4.953717231750488]], [[4.246953010559082]], [[5.275709629058838]], [[4.377523899078369]], [[4.259751319885254]], [[4.641990661621094]], [[4.629286766052246]], [[4.700592994689941]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_fecbb9c91fed1ac2177716e795bec60c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.9325298070907593]], [[1.4834176301956177]], [[1.8315191268920898]], [[1.4436335563659668]], [[1.6894466876983643]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_67784ff184ba081217645ef4cd419c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.3792216777801514]], [[2.585481643676758]], [[2.5880722999572754]], [[2.69108247756958]], [[2.578275442123413]], [[2.1945242881774902]], [[2.584146738052368]], [[2.9685192108154297]], [[2.8302712440490723]], [[2.489863634109497]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_6f144635b83f08dc0ec4e1dcb26b4f5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.49184513092041]], [[5.590744495391846]], [[5.003491401672363]], [[5.596516132354736]], [[5.173966407775879]], [[4.47603178024292]], [[4.616091251373291]], [[4.412466049194336]], [[4.7998127937316895]], [[4.498542308807373]], [[4.57667350769043]], [[5.419522762298584]], [[4.814596652984619]], [[4.749876499176025]], [[5.119980812072754]], [[5.1041579246521]], [[4.085983753204346]], [[5.195230484008789]], [[4.809571266174316]], [[4.356119632720947]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_b3221a7baf5097c19d518a4e1732e2e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.798402309417725]], [[6.471826076507568]], [[6.2061543464660645]], [[6.678011417388916]], [[6.453592777252197]], [[6.86955451965332]], [[6.800777912139893]], [[6.89926290512085]], [[7.662395000457764]], [[6.002626419067383]], [[6.689345359802246]], [[6.529068946838379]], [[6.422098159790039]], [[6.518607139587402]], [[6.604228973388672]], [[6.671124458312988]], [[6.413301467895508]], [[6.887411594390869]], [[7.293798446655273]], [[7.487094402313232]], [[6.487042427062988]], [[6.869065761566162]], [[6.210934162139893]], [[6.379401206970215]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_45049cc0fe7bd74d0630202dc45f955d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5114879608154297]], [[3.2385427951812744]], [[3.022507905960083]], [[2.5118393898010254]], [[3.1523807048797607]], [[2.381998300552368]], [[3.2308504581451416]], [[3.1900980472564697]], [[2.7136454582214355]], [[3.152656078338623]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_4f79fa1ac337074aa13ced9b0a2eea7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55f3d6566eac6d159ee25aa65d9fb14e
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.349632740020752]], [[4.581267356872559]], [[4.542279243469238]], [[4.27977180480957]], [[4.8218674659729]], [[3.832540512084961]], [[4.169091701507568]], [[4.272579669952393]], [[4.45680570602417]], [[4.216972351074219]], [[4.61435079574585]], [[4.593276023864746]], [[3.9209370613098145]], [[4.463364601135254]], [[4.558143138885498]], [[3.674036979675293]], [[4.767111778259277]], [[3.4685769081115723]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_aacba628e0cb8f59463846ee14c4e6e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c2d4786102bcc3bb26974ba10e39c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.318058967590332, 7.742279052734375, 7.49235200881958, 7.212325096130371, 7.49655294418335, 7.519100189208984, 7.2283406257629395, 7.083418369293213, 6.941357135772705, 6.601337909698486, 6.675265789031982, 7.745193004608154, 7.000194549560547, 7.5520853996276855, 7.6470723152160645, 6.704595565795898, 7.123214244842529, 8.127568244934082, 7.244419574737549, 8.030851364135742, 7.505814552307129, 7.365539073944092, 9.035940170288086, 7.207608222961426, 7.554633617401123, 6.867268085479736, 8.027496337890625, 7.540754318237305, 7.7998857498168945, 7.971572399139404]], dtype='float32').reshape([1, 30]),
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


    class TestPrimitiveOp_ce3454762b818c3099bd96e13799e686(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.729344367980957]], [[8.053105354309082]], [[7.784078598022461]], [[7.947120666503906]], [[8.634868621826172]], [[8.28966236114502]], [[7.792207717895508]], [[8.541390419006348]], [[8.528681755065918]], [[7.953497409820557]], [[8.183775901794434]], [[7.91998815536499]], [[7.978310585021973]], [[7.298711776733398]], [[8.158519744873047]], [[7.805086612701416]], [[7.681483268737793]], [[7.892937660217285]], [[8.441560745239258]], [[7.774263858795166]], [[8.654806137084961]], [[8.259748458862305]], [[7.420984745025635]], [[8.28664493560791]], [[8.260310173034668]], [[7.600845813751221]], [[8.31584644317627]], [[7.932604789733887]], [[6.945162773132324]], [[7.285499572753906]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_bebb0fdec6ca12dd1aeea35be6c6eacb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fda383f90b948933dec8cf2d32e4a8d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4133658409118652]], [[1.4126325845718384]], [[1.956082820892334]], [[1.3447970151901245]], [[1.6026768684387207]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_d635d9bc2230428858a0cbccccef0d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b494d1026b11772bb7409431868099ad
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6246941089630127]], [[2.544447898864746]], [[2.795630693435669]], [[2.9004950523376465]], [[2.514697790145874]], [[2.9879279136657715]], [[2.600862503051758]], [[2.8251161575317383]], [[2.3790998458862305]], [[2.588230609893799]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_77821b8841ff4ce8ffa59521b35ed1f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.971728324890137]], [[5.582655429840088]], [[5.47714376449585]], [[6.247722625732422]], [[5.601855754852295]], [[5.2803473472595215]], [[4.945881366729736]], [[6.030325889587402]], [[5.071672439575195]], [[5.13779354095459]], [[5.508490562438965]], [[5.381728172302246]], [[5.397087574005127]], [[4.520235061645508]], [[4.920662879943848]], [[5.529647350311279]], [[5.867205619812012]], [[5.492212295532227]], [[5.60070276260376]], [[4.916013240814209]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_10358f8aa6979ca982b32a380afc0b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c39ecb85210e2e3dd79efcdf194b69d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4f90e78a55970ab5400da30ee67ed6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665a6262b5a67a3baa6f33b4858e24c8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.0072021484375]], [[4.442929744720459]], [[4.372024059295654]], [[3.8059115409851074]], [[3.8065803050994873]], [[3.36421537399292]], [[4.133330821990967]], [[3.93577241897583]], [[4.406898498535156]], [[4.474634170532227]], [[4.070117950439453]], [[4.775428295135498]], [[4.3750386238098145]], [[3.9665915966033936]], [[3.853419065475464]], [[4.157354354858398]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_b9350e2fa5c9693363609daf9ffe8c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96dc8643dc29e249b7d4dda0732345c1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.294074535369873]], [[4.268002033233643]], [[3.913578987121582]], [[4.219417572021484]], [[4.341715335845947]], [[4.475269794464111]], [[4.291004657745361]], [[4.3809051513671875]], [[3.1958839893341064]], [[3.4503233432769775]], [[4.014822959899902]], [[3.992638111114502]], [[3.7971768379211426]], [[3.768758535385132]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_0614ba9a4db2edd697b486b57244fd6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ff5fe2aa807ad4194d0425cdbbe3a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.350203514099121]], [[4.70102071762085]], [[4.559035778045654]], [[4.141345024108887]], [[4.519817352294922]], [[4.664248466491699]], [[4.759017467498779]], [[3.98451828956604]], [[4.821098804473877]], [[4.811173915863037]], [[5.116090297698975]], [[4.81430721282959]], [[3.7403039932250977]], [[4.816918849945068]], [[4.844472885131836]], [[5.181200981140137]], [[4.626385688781738]], [[4.89996337890625]], [[4.556179523468018]], [[4.963408946990967]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_b386836ada64c9a606a30a5ef8ea538e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6fa90f351dbbfb31b9ac55f56a52360
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.00551986694336]], [[8.049155235290527]], [[7.384499549865723]], [[8.339685440063477]], [[7.984127998352051]], [[8.674389839172363]], [[8.147443771362305]], [[7.1571455001831055]], [[8.28207778930664]], [[7.843662261962891]], [[7.487427711486816]], [[7.830111503601074]], [[7.376617431640625]], [[8.312586784362793]], [[8.032049179077148]], [[8.098679542541504]], [[8.163570404052734]], [[7.175169467926025]], [[7.265573978424072]], [[7.794568061828613]], [[7.623687744140625]], [[7.449440002441406]], [[7.246952056884766]], [[7.624670028686523]], [[7.131568431854248]], [[7.504866123199463]], [[8.205310821533203]], [[8.012397766113281]], [[8.011343002319336]], [[7.762167930603027]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_4586174202d0b37009027445693ad81f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.434598922729492]], [[6.018231391906738]], [[6.290614128112793]], [[6.7121357917785645]], [[5.903967380523682]], [[5.9446611404418945]], [[5.652713298797607]], [[6.595141887664795]], [[6.70281982421875]], [[6.8267822265625]], [[7.258833885192871]], [[5.433291912078857]], [[6.281140327453613]], [[6.249874591827393]], [[5.903461456298828]], [[6.033217430114746]], [[6.7142720222473145]], [[7.203909397125244]], [[6.759840965270996]], [[6.176536560058594]], [[5.412111759185791]], [[6.212423324584961]], [[6.020208835601807]], [[6.183920860290527]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_fe681744d8a67a7374b714d395d35428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a28cf6dd4b61b0f161bcc7eb4a748b46
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.44723653793335]], [[6.102865695953369]], [[5.985477447509766]], [[6.842640399932861]], [[5.418141841888428]], [[5.740512371063232]], [[5.729429721832275]], [[6.500969886779785]], [[6.418935775756836]], [[5.616406440734863]], [[5.975831031799316]], [[5.51600456237793]], [[6.463415622711182]], [[6.480999946594238]], [[5.885873317718506]], [[5.866087436676025]], [[5.8708038330078125]], [[5.90671443939209]], [[7.097609996795654]], [[5.892422199249268]], [[6.3885416984558105]], [[5.939206123352051]], [[5.946408271789551]], [[5.895269393920898]], [[6.106153964996338]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_41586ba5a53850325c4077056e3d4c88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e113884da24056f03867a9ce2a2112a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5717320442199707]], [[3.431580066680908]], [[2.5586700439453125]], [[3.248905897140503]], [[2.9694464206695557]], [[3.5539824962615967]], [[3.0985803604125977]], [[2.924131393432617]], [[3.2537245750427246]], [[2.9115447998046875]], [[2.956376552581787]], [[3.03542423248291]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


    class TestPrimitiveOp_0edc1465fa6fbb7cf5adc55b7c2aab36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[688.7190551757812]], [[785.6703491210938]], [[735.470703125]], [[731.02197265625]], [[765.8663330078125]], [[660.8710327148438]], [[704.5062255859375]], [[700.3460693359375]], [[710.5972290039062]], [[776.5182495117188]], [[802.5549926757812]], [[751.0782470703125]], [[713.1818237304688]], [[680.6382446289062]], [[728.703857421875]], [[665.543212890625]], [[712.4335327148438]], [[682.6065063476562]], [[744.167724609375]], [[741.5267944335938]], [[755.0457763671875]], [[734.9337768554688]], [[714.0075073242188]], [[708.40283203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_8436e7be58c28ebb598dd36f5a9084c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[72.65186309814453]], [[91.7384262084961]], [[89.04048156738281]], [[86.67623138427734]], [[81.67875671386719]], [[85.5860824584961]], [[85.38785552978516]], [[87.42566680908203]], [[86.80927276611328]], [[94.9423599243164]], [[92.74808502197266]], [[80.63028717041016]], [[84.66590881347656]], [[92.82050323486328]], [[92.692138671875]], [[80.77722930908203]], [[89.87425231933594]], [[80.3315658569336]], [[93.5545654296875]], [[93.15975952148438]], [[82.20037841796875]], [[89.98966217041016]], [[84.42134857177734]], [[85.27412414550781]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_e7ce2532d5b09c397b3a3c6acd74dfbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[44.67763900756836]], [[44.78001022338867]], [[48.8891716003418]], [[47.86702346801758]], [[50.295352935791016]], [[46.41169357299805]], [[40.6933708190918]], [[45.33034133911133]], [[46.9334602355957]], [[44.16217803955078]], [[38.59510040283203]], [[49.060630798339844]], [[43.3294563293457]], [[47.460296630859375]], [[41.22036361694336]], [[42.97414016723633]], [[50.949039459228516]], [[42.96493148803711]], [[43.61009979248047]], [[46.273555755615234]], [[46.58684539794922]], [[47.14780044555664]], [[47.21940231323242]], [[48.132850646972656]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_7e088fd87126d1f004749c9be66f7ac6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[22.74772834777832]], [[23.626388549804688]], [[22.602088928222656]], [[21.93589973449707]], [[21.32398796081543]], [[22.9318904876709]], [[21.411766052246094]], [[21.45660972595215]], [[23.392732620239258]], [[23.401206970214844]], [[23.38231086730957]], [[22.44701385498047]], [[23.534521102905273]], [[21.442602157592773]], [[19.88401985168457]], [[22.459274291992188]], [[22.3023681640625]], [[21.33476448059082]], [[19.748985290527344]], [[21.4251708984375]], [[21.819591522216797]], [[22.805885314941406]], [[22.18486213684082]], [[22.38351821899414]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_5a1f7fdae3143ef07ba4450d3f6bc40a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[28815.861328125]], [[30566.94140625]], [[30025.67578125]], [[31712.501953125]], [[38763.7109375]], [[29146.6328125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_136f74435f195642aee1c4c95c3f9776(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32805.8515625]], [[37008.89453125]], [[41284.26171875]], [[31982.39453125]], [[35843.10546875]], [[39359.27734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_40daeaa0fff748e06f2e89c8e8240192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[38334.63671875]], [[42924.8359375]], [[41756.01953125]], [[38537.6953125]], [[44312.58984375]], [[51708.421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_a5fa64001841ddfba39cce47ab47c558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b638b2001d1faa1fc7472d37cf3daa9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41016.546875]], [[40018.828125]], [[41046.40625]], [[50225.1796875]], [[46902.21484375]], [[44018.38671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


    class TestPrimitiveOp_f851f8a3f993a32f136aa057f7b1bcb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a2e1cc46a0d4dd1d2fa34e6d43b3b06
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.235050678253174]], [[6.788114547729492]], [[6.863269805908203]], [[6.1629228591918945]], [[6.510852813720703]], [[6.815029144287109]], [[6.328566551208496]], [[5.321353435516357]], [[6.496885776519775]], [[6.512045860290527]], [[5.656451225280762]], [[6.609005451202393]], [[6.846685409545898]], [[6.689188480377197]], [[6.578765869140625]], [[6.044051170349121]], [[6.097584247589111]], [[6.787467956542969]], [[6.206361770629883]], [[6.546125411987305]], [[6.848434925079346]], [[6.219845771789551]], [[6.587372303009033]], [[6.3889946937561035]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_bffd315237d6162da18037bc4c07ef71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.516043186187744, 4.707422256469727, 5.164888858795166, 5.325064182281494, 5.265877723693848, 5.291617393493652, 5.1853132247924805, 5.42218017578125, 4.414474010467529, 4.513939380645752, 5.1259446144104, 4.9605937004089355, 4.702442169189453, 5.391978740692139, 5.0975661277771, 5.194991588592529, 5.173627853393555, 4.850863933563232]], dtype='float32').reshape([1, 18]),
            ]


    class TestPrimitiveOp_1eb2d19ab4708b01358c5912830b7221(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[6.707153797149658, 6.260829448699951, 6.551505088806152, 6.331302642822266, 6.301877975463867, 6.822994709014893, 6.100552558898926, 6.320962905883789, 5.698636531829834, 6.4518914222717285, 5.776687145233154, 7.555084705352783, 5.768364906311035, 6.92980432510376, 6.29964542388916, 5.982891082763672, 6.051077365875244, 6.072269916534424, 5.807013034820557, 6.226430416107178, 6.443246364593506, 6.073713302612305, 6.26926326751709]], dtype='float32').reshape([1, 23]),
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


    class TestPrimitiveOp_a82cfa57930769586016947954df49f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.477514266967773]], [[7.119456768035889]], [[7.085227012634277]], [[6.6396708488464355]], [[7.529614448547363]], [[7.339506149291992]], [[7.696058750152588]], [[7.32058572769165]], [[7.314259052276611]], [[7.595938205718994]], [[7.538951873779297]], [[6.933803081512451]], [[7.610673427581787]], [[7.8937482833862305]], [[7.015742301940918]], [[8.132213592529297]], [[7.123446941375732]], [[7.175824165344238]], [[7.174844741821289]], [[7.820660591125488]], [[7.858874797821045]], [[6.548339366912842]], [[7.482433319091797]], [[7.067334175109863]], [[7.9104390144348145]], [[6.727339267730713]], [[6.801524639129639]], [[7.090723037719727]], [[7.15854024887085]], [[7.843683242797852]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_268480a15f46dda4a0f6d95e99cb43cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.417559623718262]], [[7.743318557739258]], [[7.118884086608887]], [[7.612246036529541]], [[7.387399196624756]], [[8.145689010620117]], [[8.002328872680664]], [[7.574892520904541]], [[7.7750959396362305]], [[7.651700019836426]], [[7.327408790588379]], [[6.881934642791748]], [[8.146565437316895]], [[6.961785793304443]], [[7.403148651123047]], [[7.219143390655518]], [[7.513122081756592]], [[7.609578609466553]], [[8.403047561645508]], [[8.094966888427734]], [[7.281956672668457]], [[6.785111427307129]], [[7.188939094543457]], [[7.072811603546143]], [[7.33947229385376]], [[7.32002592086792]], [[7.809920787811279]], [[7.630326747894287]], [[7.602578163146973]], [[7.1075544357299805]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_e100b13dd610a7f8edd1473fbce2a8a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28c951b35293a433ff6d8dc863996479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.369239330291748]], [[1.2205345630645752]], [[1.3513569831848145]], [[1.3422176837921143]], [[1.3516736030578613]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_6530294bec70ceae3563e37d3c05ca8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.545330047607422]], [[2.542398691177368]], [[2.236086130142212]], [[2.722609519958496]], [[2.6072356700897217]], [[3.208588123321533]], [[3.3306355476379395]], [[2.2302825450897217]], [[2.476623058319092]], [[2.767191171646118]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f56e7ab9c43d0a63758266c6d5502e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.671944618225098]], [[6.098160743713379]], [[5.53664493560791]], [[5.733607292175293]], [[6.2740702629089355]], [[5.394620895385742]], [[5.611918926239014]], [[5.351341724395752]], [[6.054460525512695]], [[5.389400959014893]], [[5.4167094230651855]], [[5.1283745765686035]], [[5.846133232116699]], [[5.851635456085205]], [[5.39577579498291]], [[6.457047462463379]], [[6.039535045623779]], [[6.229255676269531]], [[5.209201335906982]], [[5.494519233703613]], [[6.435673236846924]], [[6.3030104637146]], [[4.4522929191589355]], [[6.078214645385742]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_9118bbfa906b08b71048a3808f7e281e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.120477676391602]], [[4.59541130065918]], [[4.452274799346924]], [[3.8959805965423584]], [[4.421902179718018]], [[4.485260963439941]], [[5.113558769226074]], [[4.413151264190674]], [[4.051578521728516]], [[4.178587436676025]], [[4.9391093254089355]], [[4.186689853668213]], [[4.489201068878174]], [[4.360198497772217]], [[4.403806686401367]], [[4.735654354095459]], [[4.867974281311035]], [[4.216265678405762]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_8ebda7ae6be1a5cdddfea8c06368baa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8fa68c6d39f0fc248526a9d30867a3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.3108625411987305]], [[6.79969596862793]], [[6.751509666442871]], [[6.1104865074157715]], [[6.556639194488525]], [[6.563843250274658]], [[6.04982328414917]], [[6.986202716827393]], [[6.39040470123291]], [[6.591248035430908]], [[6.4414215087890625]], [[6.0124945640563965]], [[6.929512977600098]], [[6.34401798248291]], [[7.0286431312561035]], [[6.831506252288818]], [[5.958920955657959]], [[5.804043292999268]], [[5.180810928344727]], [[5.40272331237793]], [[6.816746711730957]], [[5.976834774017334]], [[6.840426445007324]], [[6.716890811920166]]]], dtype='float32').reshape([1, 24, 1, 1]),
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


    class TestPrimitiveOp_865e3ef989c55f384bae6e03112fda61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8333555459976196]], [[0.4858071208000183]], [[1.1957030296325684]], [[1.2955039739608765]]]], dtype='float32').reshape([1, 4, 1, 1]),
            ]


    class TestPrimitiveOp_661b800fa7b4dcf6c02d320a8849130c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a989055ba13801cd14cd2caf6ddea643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5765128135681152]], [[2.6508169174194336]], [[2.5669305324554443]], [[2.265747547149658]], [[2.859575033187866]], [[3.018110752105713]], [[2.4999642372131348]], [[2.6135239601135254]], [[2.8138022422790527]], [[2.400343894958496]], [[2.293578863143921]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_03420c446c45f27061fcb3e28c432e4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.678655624389648]], [[7.421043872833252]], [[7.546217441558838]], [[7.017660140991211]], [[7.533844947814941]], [[7.912869453430176]], [[7.165874004364014]], [[7.757705211639404]], [[7.679967880249023]], [[7.868678569793701]], [[7.383896350860596]], [[7.19304895401001]], [[7.269775390625]], [[7.78610897064209]], [[7.44060754776001]], [[7.792629241943359]], [[6.972398281097412]], [[7.668750286102295]], [[7.87344217300415]], [[7.160478591918945]], [[7.612797737121582]], [[7.538360118865967]], [[7.100478172302246]], [[7.216472148895264]], [[7.498114109039307]], [[7.365005970001221]], [[7.5870442390441895]], [[7.954311847686768]], [[6.997818470001221]], [[6.68562650680542]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_d1255b3f1dc606f5ef6e1700d0f461e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.204582214355469]], [[4.17450475692749]], [[4.450677394866943]], [[5.001364707946777]], [[4.378679275512695]], [[4.139589786529541]], [[4.008517742156982]], [[4.55598258972168]], [[4.8348588943481445]], [[3.643404006958008]], [[5.065579891204834]], [[4.885112285614014]], [[4.120347499847412]], [[4.529646873474121]], [[3.988499879837036]], [[3.9207963943481445]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_7cc5a60fed614c305abd10276f226af5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.332149505615234]], [[8.124646186828613]], [[9.008127212524414]], [[7.852988243103027]], [[7.852867126464844]], [[7.848668575286865]], [[8.38390064239502]], [[7.666966915130615]], [[8.083961486816406]], [[8.015180587768555]], [[8.18043327331543]], [[7.053821086883545]], [[7.988963603973389]], [[7.258120536804199]], [[7.686677932739258]], [[7.7280659675598145]], [[7.685739040374756]], [[7.41143274307251]], [[6.809802532196045]], [[8.155046463012695]], [[6.916194438934326]], [[7.524030685424805]], [[8.538736343383789]], [[7.1718244552612305]], [[7.731924057006836]], [[7.71543550491333]], [[7.578956604003906]], [[7.180920124053955]], [[7.5848236083984375]], [[7.148171424865723]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_6d3f7221de44d20f64aeaed3bc3663b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.464395046234131]], [[6.970254421234131]], [[6.331833839416504]], [[6.208786487579346]], [[6.412256240844727]], [[6.44594144821167]], [[6.668659687042236]], [[5.506393909454346]], [[5.857957363128662]], [[6.159074783325195]], [[6.215778350830078]], [[7.449837684631348]], [[6.134645462036133]], [[7.1130499839782715]], [[6.447407245635986]], [[6.364443778991699]], [[6.062101364135742]], [[6.829754829406738]], [[6.665282249450684]], [[5.587331295013428]], [[6.970404624938965]], [[6.038416385650635]], [[6.559352397918701]], [[7.202384948730469]], [[6.573240756988525]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_549ebeb5771d45001b1545c8a27adbd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.648985385894775]], [[4.977217197418213]], [[4.969912052154541]], [[4.990406513214111]], [[5.3486456871032715]], [[5.269417762756348]], [[5.3413591384887695]], [[4.685579776763916]], [[4.341116905212402]], [[5.372086524963379]], [[4.675849914550781]], [[5.339875221252441]], [[4.787026405334473]], [[4.957879066467285]], [[4.992903709411621]], [[4.947808742523193]], [[4.469698905944824]], [[3.9869487285614014]], [[4.38286018371582]], [[5.121321201324463]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_87dd132a4c1729d655c799710904fa7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.745029449462891]], [[4.362461090087891]], [[4.826979160308838]], [[4.8406877517700195]], [[4.93057918548584]], [[4.475962162017822]], [[4.890925407409668]], [[4.360519886016846]], [[5.017579555511475]], [[5.275971412658691]], [[4.956449508666992]], [[4.257177829742432]], [[4.3468146324157715]], [[4.757590293884277]], [[4.607741355895996]], [[4.7346930503845215]], [[5.006930828094482]], [[5.1289238929748535]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_f9a84dbc1b2a84f01b3fa26a8664154e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.018347263336182]], [[5.169200897216797]], [[4.998176097869873]], [[4.5034708976745605]], [[4.442930221557617]], [[5.493774890899658]], [[5.000889301300049]], [[4.583656311035156]], [[5.095301628112793]], [[5.118431568145752]], [[4.825158596038818]], [[4.5962653160095215]], [[4.8283796310424805]], [[4.349386215209961]], [[4.9702229499816895]], [[4.948547840118408]], [[4.366069793701172]], [[4.897094249725342]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_985f2e0fdb496dd46b3239326dfa0c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15b5c3ab70a6d6f25a85c80f6470bc21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.762831211090088]], [[5.912539482116699]], [[6.511545181274414]], [[5.992820739746094]], [[6.968940258026123]], [[6.612647533416748]], [[6.161623001098633]], [[5.722076892852783]], [[6.703250408172607]], [[5.690288066864014]], [[6.0369768142700195]], [[6.201291084289551]], [[6.706302642822266]], [[5.62612771987915]], [[6.245187759399414]], [[5.9466352462768555]], [[5.768355846405029]], [[6.4490766525268555]], [[5.810674667358398]], [[5.889484405517578]], [[6.537364959716797]], [[5.689944744110107]], [[6.338402271270752]], [[7.280892372131348]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_030909805c34712c06a25bccb75b4358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619fae6809110a69c1e44716186e0368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.857046127319336]], [[4.94091796875]], [[4.483896732330322]], [[3.889404535293579]], [[4.412776947021484]], [[3.7807023525238037]], [[3.8521409034729004]], [[4.523113250732422]], [[4.566526889801025]], [[3.31935453414917]], [[4.453447341918945]], [[4.488199234008789]], [[5.0645952224731445]], [[4.404751300811768]], [[5.043583869934082]], [[4.1073713302612305]], [[4.769421100616455]], [[4.5601487159729]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_544dce1b212f6500f751ab424c549180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.269239902496338]], [[6.215242385864258]], [[4.777440071105957]], [[5.51981258392334]], [[5.698533535003662]], [[5.038366794586182]], [[6.198485851287842]], [[5.795487403869629]], [[5.765268325805664]], [[6.254849433898926]], [[6.3785247802734375]], [[5.870089530944824]], [[5.322088241577148]], [[5.806186676025391]], [[5.142665863037109]], [[5.517502307891846]], [[6.4795145988464355]], [[5.786216735839844]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_3e3c79781e3e7c77ab64520ab94e3063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.686263084411621]], [[4.615196704864502]], [[4.329402446746826]], [[5.435118675231934]], [[4.637634754180908]], [[4.600268363952637]], [[5.243618965148926]], [[4.374950885772705]], [[4.660830974578857]], [[4.971651554107666]], [[4.599851608276367]], [[4.9320197105407715]], [[4.223817348480225]], [[4.908278465270996]], [[5.701253890991211]], [[5.603471755981445]], [[5.186424732208252]], [[5.014660358428955]]]], dtype='float32').reshape([1, 18, 1, 1]),
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


    class TestPrimitiveOp_f8c907f5d658bcb22d050e2c117cb187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.9503366947174072]], [[3.9157907962799072]], [[4.34525203704834]], [[4.444269180297852]], [[4.11352014541626]], [[4.425756931304932]], [[4.358208179473877]], [[4.3368682861328125]], [[3.8690223693847656]], [[4.649980545043945]], [[4.770483493804932]], [[4.135887622833252]], [[3.9279897212982178]], [[4.586308479309082]], [[4.425254821777344]], [[5.121748924255371]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_9aca422e1aec58152865d589082e427d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.872812271118164]], [[4.813311576843262]], [[4.442329406738281]], [[4.180283546447754]], [[4.563350677490234]], [[4.228987693786621]], [[5.5067973136901855]], [[4.991118431091309]], [[5.469857215881348]], [[4.687140464782715]], [[5.060680389404297]], [[4.81998348236084]], [[4.570867538452148]], [[4.406635284423828]], [[5.396178722381592]], [[4.410801887512207]], [[5.008913040161133]], [[5.029808044433594]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_5791888bf6de5934ffcb1af7baf9d884(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.8795795440673828]], [[1.038812518119812]], [[0.8776819109916687]], [[1.1465747356414795]]]], dtype='float32').reshape([1, 4, 1, 1]),
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


    class TestPrimitiveOp_2f8057146ee155e766838342522b493f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.57105016708374]], [[5.187228679656982]], [[4.687733173370361]], [[4.013662338256836]], [[5.9551472663879395]], [[5.180907726287842]], [[5.969728469848633]], [[5.038147926330566]], [[5.276065349578857]], [[5.582442283630371]], [[5.576443672180176]], [[5.48445987701416]], [[5.050901889801025]], [[4.763657569885254]], [[5.620220184326172]], [[4.300808906555176]], [[5.081611633300781]], [[5.307088375091553]], [[5.391952037811279]], [[5.492855548858643]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_e341740e0f212bca7cbc0355e70c8765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_242a60417ccc5520b5683f6d11f4664d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.047842025756836]], [[3.1700327396392822]], [[3.9174647331237793]], [[3.7232398986816406]], [[3.2468199729919434]], [[4.048626899719238]], [[3.1338746547698975]], [[3.322093963623047]], [[3.064919948577881]], [[3.4265213012695312]], [[3.5901713371276855]], [[3.7378084659576416]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_b60136f48a9e63cd0eff794e23878e0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.73850679397583]], [[4.863675594329834]], [[5.502257347106934]], [[4.718661308288574]], [[5.080923080444336]], [[4.2703328132629395]], [[5.199297904968262]], [[5.081809997558594]], [[4.6135382652282715]], [[4.608609676361084]], [[5.072863578796387]], [[4.551656246185303]], [[5.205888748168945]], [[4.730347633361816]], [[4.898443222045898]], [[5.186558246612549]], [[5.368105411529541]], [[5.535336017608643]], [[5.157254219055176]], [[5.266867637634277]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_17dc9536014d0660fed4f975e940f238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.506019115447998]], [[2.3050897121429443]], [[2.5983593463897705]], [[2.21671199798584]], [[2.4672653675079346]], [[2.830336570739746]], [[2.7277143001556396]], [[2.5625483989715576]], [[2.609344959259033]], [[2.6353414058685303]], [[2.517425060272217]]]], dtype='float32').reshape([1, 11, 1, 1]),
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


    class TestPrimitiveOp_614584409395087cda0ee2f186d72bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.322866916656494]], [[4.614137172698975]], [[3.6140758991241455]], [[3.754441261291504]], [[3.403918743133545]], [[3.782527446746826]], [[3.539771556854248]], [[3.5740444660186768]], [[4.168973922729492]], [[4.086240291595459]], [[4.050668239593506]], [[3.755788564682007]], [[3.9075751304626465]], [[3.614896297454834]]]], dtype='float32').reshape([1, 14, 1, 1]),
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


    class TestPrimitiveOp_832249c72908de999c9f1a996e7bfece(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.513790130615234]], [[4.512863636016846]], [[5.205973148345947]], [[5.517946243286133]], [[4.983819961547852]], [[6.141815185546875]], [[5.8559980392456055]], [[5.97186279296875]], [[6.138519763946533]], [[5.829843997955322]], [[5.704123497009277]], [[5.654500961303711]], [[5.876495361328125]], [[5.6354475021362305]], [[5.825729846954346]], [[5.408461093902588]], [[5.917852401733398]], [[5.67879581451416]], [[5.08447790145874]], [[5.052452087402344]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_87b976aa86042da642a604f83b8cd1ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[34324.3359375]], [[35210.25]], [[35337.359375]], [[27557.31640625]], [[37704.00390625]], [[39159.00390625]]], [[[35315.1015625]], [[36224.921875]], [[36351.7421875]], [[28357.1328125]], [[38788.0546875]], [[40294.0859375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_f8a2f3da0f0588a3603f7ba6f3993b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[33221.203125]], [[39309.6171875]], [[33714.2109375]], [[35410.09765625]], [[41504.390625]], [[40679.7265625]]], [[[33992.7578125]], [[40221.5]], [[34499.7421875]], [[36229.44921875]], [[42466.921875]], [[41625.83984375]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_5f4b821020d995e7578a833fb8fc038c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[43299.49609375]], [[46142.76171875]], [[36165.4921875]], [[34835.8125]], [[49133.00390625]], [[27065.525390625]]], [[[44534.171875]], [[47459.6953125]], [[37196.3203125]], [[35827.3828125]], [[50539.9609375]], [[27837.369140625]]]], dtype='float32').reshape([2, 6, 1, 1]),
            ]


    class TestPrimitiveOp_d0ca916bb2645407e7d8122e4370f3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32312.3125]], [[36691.55859375]], [[38778.1171875]], [[42773.34765625]], [[37369.8203125]], [[45978.91796875]]], [[[33218.5703125]], [[37730.0859375]], [[39871.8984375]], [[43983.71875]], [[38432.6796875]], [[47276.96875]]]], dtype='float32').reshape([2, 6, 1, 1]),
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


    class TestPrimitiveOp_a1e8d39506e4774a4a631d0fb6dbeff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.427326679229736]], [[7.228569030761719]], [[6.992318153381348]], [[6.183574199676514]], [[7.347770690917969]], [[7.364208698272705]], [[7.418874740600586]], [[7.159008026123047]], [[6.773091793060303]], [[6.469487190246582]], [[7.069914817810059]], [[7.183864593505859]], [[6.996612548828125]], [[6.526675224304199]], [[7.3109564781188965]], [[7.045126438140869]], [[7.064694404602051]], [[6.389091491699219]], [[6.853155612945557]], [[7.308903694152832]], [[6.340528964996338]], [[6.342247009277344]], [[7.179606914520264]], [[7.434704780578613]], [[6.474822521209717]], [[7.299975872039795]], [[6.992025375366211]], [[7.441059112548828]], [[7.8285698890686035]], [[7.170892715454102]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_0b9c40b6bd7724b2bfc89a223323d038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.369341850280762]], [[7.440235137939453]], [[7.959526062011719]], [[6.732922554016113]], [[7.920384883880615]], [[6.782915115356445]], [[6.6313252449035645]], [[6.644609451293945]], [[8.01773738861084]], [[7.6204633712768555]], [[6.5885396003723145]], [[7.120371341705322]], [[7.276170253753662]], [[7.940833568572998]], [[7.641347885131836]], [[7.419842720031738]], [[7.180537223815918]], [[7.764443397521973]], [[6.988077163696289]], [[6.984099864959717]], [[6.495869159698486]], [[6.951446533203125]], [[7.682574272155762]], [[7.8235368728637695]], [[7.526726722717285]], [[7.461899757385254]], [[6.840702056884766]], [[7.3826093673706055]], [[7.137312889099121]], [[7.787014007568359]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_b5b16d26e52c291dbc628fe7ba3e2eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf445f6fcb7e1cf6906b8d3320c954b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.031045913696289]], [[8.570387840270996]], [[7.079840660095215]], [[6.9657511711120605]], [[8.437033653259277]], [[7.539844989776611]], [[7.698031902313232]], [[7.678504943847656]], [[6.655674934387207]], [[6.759970188140869]], [[7.527650356292725]], [[7.25475549697876]], [[7.833059310913086]], [[7.854125022888184]], [[6.990649223327637]], [[7.532878875732422]], [[8.42581558227539]], [[7.625922679901123]], [[7.514565467834473]], [[8.335790634155273]], [[7.531124114990234]], [[7.539545059204102]], [[7.470608234405518]], [[7.559680461883545]], [[7.1960554122924805]], [[7.70048189163208]], [[7.439781188964844]], [[7.400554180145264]], [[8.132777214050293]], [[7.975470066070557]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_d5f4214f90e2613bd2ea0ff7c37bfc76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.580674648284912]], [[7.62647008895874]], [[6.839269638061523]], [[7.1614837646484375]], [[8.016974449157715]], [[6.909478187561035]], [[7.518780708312988]], [[6.050467491149902]], [[7.179502010345459]], [[7.222623348236084]], [[7.504827976226807]], [[7.768027305603027]], [[7.591369152069092]], [[7.284604549407959]], [[7.636691570281982]], [[6.545493125915527]], [[7.5735344886779785]], [[7.230812072753906]], [[6.924068450927734]], [[7.332103252410889]], [[7.459962368011475]], [[7.344234466552734]], [[7.400725364685059]], [[6.7071123123168945]], [[7.448028087615967]], [[7.550012588500977]], [[6.99104642868042]], [[7.519863128662109]], [[8.079681396484375]], [[7.346412658691406]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_0045ed834ce180e1627afab19c7c7006(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.3224000930786133]], [[2.8902981281280518]], [[2.7788562774658203]], [[3.364142894744873]], [[3.5580146312713623]], [[2.896169900894165]], [[3.005430221557617]], [[3.2257089614868164]], [[2.7983479499816895]], [[2.645906686782837]], [[3.040734052658081]], [[3.4453206062316895]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_7e490be208883b49886f5f071c8c0159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.9385769367218018]], [[2.4369940757751465]], [[3.0526251792907715]], [[2.676270008087158]], [[3.005340337753296]], [[2.3457677364349365]], [[2.8266632556915283]], [[3.110970973968506]], [[2.641709566116333]], [[2.2739691734313965]], [[2.794102191925049]], [[3.12825083732605]]]], dtype='float32').reshape([1, 12, 1, 1]),
            ]


    class TestPrimitiveOp_ce17d6bd9324bb155d495f25fc599ee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.661965370178223]], [[6.609972953796387]], [[5.915094375610352]], [[6.050534725189209]], [[6.65255880355835]], [[5.39961051940918]], [[7.225243091583252]], [[6.373495101928711]], [[5.898714542388916]], [[6.293486595153809]], [[6.69158411026001]], [[5.537364482879639]], [[5.7546796798706055]], [[6.335968017578125]], [[6.212339401245117]], [[6.0673675537109375]], [[6.005062103271484]], [[5.6984543800354]], [[6.54160737991333]], [[6.147066593170166]], [[6.136723041534424]], [[6.483702182769775]], [[6.854306221008301]], [[6.514199733734131]], [[6.482630252838135]]]], dtype='float32').reshape([1, 25, 1, 1]),
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


    class TestPrimitiveOp_58913795ff0011ae5295b5b61af3d175(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.467135906219482]], [[4.3664231300354]], [[4.372544288635254]], [[4.080681324005127]], [[4.4120869636535645]], [[4.544767379760742]], [[4.381662368774414]], [[4.579371929168701]], [[4.712819576263428]], [[5.40225076675415]], [[4.953717231750488]], [[4.246953010559082]], [[5.275709629058838]], [[4.377523899078369]], [[4.259751319885254]], [[4.641990661621094]], [[4.629286766052246]], [[4.700592994689941]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_6480f898595034dc31697d642337f1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c79b9b7c48e4b700153863d730cee29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.9325298070907593]], [[1.4834176301956177]], [[1.8315191268920898]], [[1.4436335563659668]], [[1.6894466876983643]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_7676e265a315bf3ff803af91c3e16617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.3792216777801514]], [[2.585481643676758]], [[2.5880722999572754]], [[2.69108247756958]], [[2.578275442123413]], [[2.1945242881774902]], [[2.584146738052368]], [[2.9685192108154297]], [[2.8302712440490723]], [[2.489863634109497]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_4ed57dabae36ddf2b2dee5fd861baa45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.49184513092041]], [[5.590744495391846]], [[5.003491401672363]], [[5.596516132354736]], [[5.173966407775879]], [[4.47603178024292]], [[4.616091251373291]], [[4.412466049194336]], [[4.7998127937316895]], [[4.498542308807373]], [[4.57667350769043]], [[5.419522762298584]], [[4.814596652984619]], [[4.749876499176025]], [[5.119980812072754]], [[5.1041579246521]], [[4.085983753204346]], [[5.195230484008789]], [[4.809571266174316]], [[4.356119632720947]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_03fe1b002e82700c911466059b6f3776(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.798402309417725]], [[6.471826076507568]], [[6.2061543464660645]], [[6.678011417388916]], [[6.453592777252197]], [[6.86955451965332]], [[6.800777912139893]], [[6.89926290512085]], [[7.662395000457764]], [[6.002626419067383]], [[6.689345359802246]], [[6.529068946838379]], [[6.422098159790039]], [[6.518607139587402]], [[6.604228973388672]], [[6.671124458312988]], [[6.413301467895508]], [[6.887411594390869]], [[7.293798446655273]], [[7.487094402313232]], [[6.487042427062988]], [[6.869065761566162]], [[6.210934162139893]], [[6.379401206970215]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_2fe93505efa16cf7c1067f074d59de7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_021dc135d79b7aefd4791e6e6515cc9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5114879608154297]], [[3.2385427951812744]], [[3.022507905960083]], [[2.5118393898010254]], [[3.1523807048797607]], [[2.381998300552368]], [[3.2308504581451416]], [[3.1900980472564697]], [[2.7136454582214355]], [[3.152656078338623]]]], dtype='float32').reshape([1, 10, 1, 1]),
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


    class TestPrimitiveOp_71599fefff1d97b1700407287c35265b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.349632740020752]], [[4.581267356872559]], [[4.542279243469238]], [[4.27977180480957]], [[4.8218674659729]], [[3.832540512084961]], [[4.169091701507568]], [[4.272579669952393]], [[4.45680570602417]], [[4.216972351074219]], [[4.61435079574585]], [[4.593276023864746]], [[3.9209370613098145]], [[4.463364601135254]], [[4.558143138885498]], [[3.674036979675293]], [[4.767111778259277]], [[3.4685769081115723]]]], dtype='float32').reshape([1, 18, 1, 1]),
            ]


    class TestPrimitiveOp_8d0098034105ed6cd96385ea5d8e4cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_625806210b7bc905ec11eef79a288dca
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.318058967590332, 7.742279052734375, 7.49235200881958, 7.212325096130371, 7.49655294418335, 7.519100189208984, 7.2283406257629395, 7.083418369293213, 6.941357135772705, 6.601337909698486, 6.675265789031982, 7.745193004608154, 7.000194549560547, 7.5520853996276855, 7.6470723152160645, 6.704595565795898, 7.123214244842529, 8.127568244934082, 7.244419574737549, 8.030851364135742, 7.505814552307129, 7.365539073944092, 9.035940170288086, 7.207608222961426, 7.554633617401123, 6.867268085479736, 8.027496337890625, 7.540754318237305, 7.7998857498168945, 7.971572399139404]], dtype='float32').reshape([1, 30]),
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


    class TestPrimitiveOp_596950f75f3202e0487ba4690a60f9f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.729344367980957]], [[8.053105354309082]], [[7.784078598022461]], [[7.947120666503906]], [[8.634868621826172]], [[8.28966236114502]], [[7.792207717895508]], [[8.541390419006348]], [[8.528681755065918]], [[7.953497409820557]], [[8.183775901794434]], [[7.91998815536499]], [[7.978310585021973]], [[7.298711776733398]], [[8.158519744873047]], [[7.805086612701416]], [[7.681483268737793]], [[7.892937660217285]], [[8.441560745239258]], [[7.774263858795166]], [[8.654806137084961]], [[8.259748458862305]], [[7.420984745025635]], [[8.28664493560791]], [[8.260310173034668]], [[7.600845813751221]], [[8.31584644317627]], [[7.932604789733887]], [[6.945162773132324]], [[7.285499572753906]]]], dtype='float32').reshape([1, 30, 1, 1]),
            ]


    class TestPrimitiveOp_9875d11a15196d68fe3ff5805228ecfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4133658409118652]], [[1.4126325845718384]], [[1.956082820892334]], [[1.3447970151901245]], [[1.6026768684387207]]]], dtype='float32').reshape([1, 5, 1, 1]),
            ]


    class TestPrimitiveOp_48bfd3fab6d7a425d602aa8fd102dd1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.6246941089630127]], [[2.544447898864746]], [[2.795630693435669]], [[2.9004950523376465]], [[2.514697790145874]], [[2.9879279136657715]], [[2.600862503051758]], [[2.8251161575317383]], [[2.3790998458862305]], [[2.588230609893799]]]], dtype='float32').reshape([1, 10, 1, 1]),
            ]


    class TestPrimitiveOp_a96e26bc85fb1fe2570e58327a96d4fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.971728324890137]], [[5.582655429840088]], [[5.47714376449585]], [[6.247722625732422]], [[5.601855754852295]], [[5.2803473472595215]], [[4.945881366729736]], [[6.030325889587402]], [[5.071672439575195]], [[5.13779354095459]], [[5.508490562438965]], [[5.381728172302246]], [[5.397087574005127]], [[4.520235061645508]], [[4.920662879943848]], [[5.529647350311279]], [[5.867205619812012]], [[5.492212295532227]], [[5.60070276260376]], [[4.916013240814209]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_a33122bedb8477240e3a1c119450491a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_961145cd99c7e58d22bc6d254fd948ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[4.0072021484375]], [[4.442929744720459]], [[4.372024059295654]], [[3.8059115409851074]], [[3.8065803050994873]], [[3.36421537399292]], [[4.133330821990967]], [[3.93577241897583]], [[4.406898498535156]], [[4.474634170532227]], [[4.070117950439453]], [[4.775428295135498]], [[4.3750386238098145]], [[3.9665915966033936]], [[3.853419065475464]], [[4.157354354858398]]]], dtype='float32').reshape([1, 16, 1, 1]),
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


    class TestPrimitiveOp_51502ce2d60ae524f138e947c914125f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[3.294074535369873]], [[4.268002033233643]], [[3.913578987121582]], [[4.219417572021484]], [[4.341715335845947]], [[4.475269794464111]], [[4.291004657745361]], [[4.3809051513671875]], [[3.1958839893341064]], [[3.4503233432769775]], [[4.014822959899902]], [[3.992638111114502]], [[3.7971768379211426]], [[3.768758535385132]]]], dtype='float32').reshape([1, 14, 1, 1]),
            ]


    class TestPrimitiveOp_812baf03bb8f02a384e2820b83ef39ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[5.350203514099121]], [[4.70102071762085]], [[4.559035778045654]], [[4.141345024108887]], [[4.519817352294922]], [[4.664248466491699]], [[4.759017467498779]], [[3.98451828956604]], [[4.821098804473877]], [[4.811173915863037]], [[5.116090297698975]], [[4.81430721282959]], [[3.7403039932250977]], [[4.816918849945068]], [[4.844472885131836]], [[5.181200981140137]], [[4.626385688781738]], [[4.89996337890625]], [[4.556179523468018]], [[4.963408946990967]]]], dtype='float32').reshape([1, 20, 1, 1]),
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


    class TestPrimitiveOp_0b68ff6b92e0e303ee06aa00a1e7903a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[8.00551986694336]], [[8.049155235290527]], [[7.384499549865723]], [[8.339685440063477]], [[7.984127998352051]], [[8.674389839172363]], [[8.147443771362305]], [[7.1571455001831055]], [[8.28207778930664]], [[7.843662261962891]], [[7.487427711486816]], [[7.830111503601074]], [[7.376617431640625]], [[8.312586784362793]], [[8.032049179077148]], [[8.098679542541504]], [[8.163570404052734]], [[7.175169467926025]], [[7.265573978424072]], [[7.794568061828613]], [[7.623687744140625]], [[7.449440002441406]], [[7.246952056884766]], [[7.624670028686523]], [[7.131568431854248]], [[7.504866123199463]], [[8.205310821533203]], [[8.012397766113281]], [[8.011343002319336]], [[7.762167930603027]]]], dtype='float32').reshape([1, 30, 1, 1]),
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


    class TestPrimitiveOp_78b61f50d9b399030f0e7bddfce6e8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.434598922729492]], [[6.018231391906738]], [[6.290614128112793]], [[6.7121357917785645]], [[5.903967380523682]], [[5.9446611404418945]], [[5.652713298797607]], [[6.595141887664795]], [[6.70281982421875]], [[6.8267822265625]], [[7.258833885192871]], [[5.433291912078857]], [[6.281140327453613]], [[6.249874591827393]], [[5.903461456298828]], [[6.033217430114746]], [[6.7142720222473145]], [[7.203909397125244]], [[6.759840965270996]], [[6.176536560058594]], [[5.412111759185791]], [[6.212423324584961]], [[6.020208835601807]], [[6.183920860290527]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_23d49001262b83c044987a36bcec29a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[6.44723653793335]], [[6.102865695953369]], [[5.985477447509766]], [[6.842640399932861]], [[5.418141841888428]], [[5.740512371063232]], [[5.729429721832275]], [[6.500969886779785]], [[6.418935775756836]], [[5.616406440734863]], [[5.975831031799316]], [[5.51600456237793]], [[6.463415622711182]], [[6.480999946594238]], [[5.885873317718506]], [[5.866087436676025]], [[5.8708038330078125]], [[5.90671443939209]], [[7.097609996795654]], [[5.892422199249268]], [[6.3885416984558105]], [[5.939206123352051]], [[5.946408271789551]], [[5.895269393920898]], [[6.106153964996338]]]], dtype='float32').reshape([1, 25, 1, 1]),
            ]


    class TestPrimitiveOp_8e1507b404f0c1971afba08fd67af93d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.5717320442199707]], [[3.431580066680908]], [[2.5586700439453125]], [[3.248905897140503]], [[2.9694464206695557]], [[3.5539824962615967]], [[3.0985803604125977]], [[2.924131393432617]], [[3.2537245750427246]], [[2.9115447998046875]], [[2.956376552581787]], [[3.03542423248291]]]], dtype='float32').reshape([1, 12, 1, 1]),
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


    class TestPrimitiveOp_9e66a39e94c2ea9ee3cbb9ae155fa377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[688.7190551757812]], [[785.6703491210938]], [[735.470703125]], [[731.02197265625]], [[765.8663330078125]], [[660.8710327148438]], [[704.5062255859375]], [[700.3460693359375]], [[710.5972290039062]], [[776.5182495117188]], [[802.5549926757812]], [[751.0782470703125]], [[713.1818237304688]], [[680.6382446289062]], [[728.703857421875]], [[665.543212890625]], [[712.4335327148438]], [[682.6065063476562]], [[744.167724609375]], [[741.5267944335938]], [[755.0457763671875]], [[734.9337768554688]], [[714.0075073242188]], [[708.40283203125]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_ef47b66c193754dfc994fa6aad90fbc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[72.65186309814453]], [[91.7384262084961]], [[89.04048156738281]], [[86.67623138427734]], [[81.67875671386719]], [[85.5860824584961]], [[85.38785552978516]], [[87.42566680908203]], [[86.80927276611328]], [[94.9423599243164]], [[92.74808502197266]], [[80.63028717041016]], [[84.66590881347656]], [[92.82050323486328]], [[92.692138671875]], [[80.77722930908203]], [[89.87425231933594]], [[80.3315658569336]], [[93.5545654296875]], [[93.15975952148438]], [[82.20037841796875]], [[89.98966217041016]], [[84.42134857177734]], [[85.27412414550781]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_7f33139fcceb10ec581cd58801a1565b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[44.67763900756836]], [[44.78001022338867]], [[48.8891716003418]], [[47.86702346801758]], [[50.295352935791016]], [[46.41169357299805]], [[40.6933708190918]], [[45.33034133911133]], [[46.9334602355957]], [[44.16217803955078]], [[38.59510040283203]], [[49.060630798339844]], [[43.3294563293457]], [[47.460296630859375]], [[41.22036361694336]], [[42.97414016723633]], [[50.949039459228516]], [[42.96493148803711]], [[43.61009979248047]], [[46.273555755615234]], [[46.58684539794922]], [[47.14780044555664]], [[47.21940231323242]], [[48.132850646972656]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_c73e6e53338ead891d2db38b207f9224(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[22.74772834777832]], [[23.626388549804688]], [[22.602088928222656]], [[21.93589973449707]], [[21.32398796081543]], [[22.9318904876709]], [[21.411766052246094]], [[21.45660972595215]], [[23.392732620239258]], [[23.401206970214844]], [[23.38231086730957]], [[22.44701385498047]], [[23.534521102905273]], [[21.442602157592773]], [[19.88401985168457]], [[22.459274291992188]], [[22.3023681640625]], [[21.33476448059082]], [[19.748985290527344]], [[21.4251708984375]], [[21.819591522216797]], [[22.805885314941406]], [[22.18486213684082]], [[22.38351821899414]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_252c2b0bef875425a344697f10f7e519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[28815.861328125]], [[30566.94140625]], [[30025.67578125]], [[31712.501953125]], [[38763.7109375]], [[29146.6328125]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_dcef22366d3ad38d7b8113589f27e859(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[32805.8515625]], [[37008.89453125]], [[41284.26171875]], [[31982.39453125]], [[35843.10546875]], [[39359.27734375]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_34091f887998391169e2babb298a1d01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[38334.63671875]], [[42924.8359375]], [[41756.01953125]], [[38537.6953125]], [[44312.58984375]], [[51708.421875]]]], dtype='float32').reshape([1, 6, 1, 1]),
            ]


    class TestPrimitiveOp_824ea283a398c69dcc842cc21161642d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[41016.546875]], [[40018.828125]], [[41046.40625]], [[50225.1796875]], [[46902.21484375]], [[44018.38671875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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


    class TestPrimitiveOp_aecce483b1fd17004c4d7df6e5b5da68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35113bc0e82d84882f705d675eda32a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[7.235050678253174]], [[6.788114547729492]], [[6.863269805908203]], [[6.1629228591918945]], [[6.510852813720703]], [[6.815029144287109]], [[6.328566551208496]], [[5.321353435516357]], [[6.496885776519775]], [[6.512045860290527]], [[5.656451225280762]], [[6.609005451202393]], [[6.846685409545898]], [[6.689188480377197]], [[6.578765869140625]], [[6.044051170349121]], [[6.097584247589111]], [[6.787467956542969]], [[6.206361770629883]], [[6.546125411987305]], [[6.848434925079346]], [[6.219845771789551]], [[6.587372303009033]], [[6.3889946937561035]]]], dtype='float32').reshape([1, 24, 1, 1]),
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