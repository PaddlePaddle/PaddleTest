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
    class PrimitiveOp_054249f0a2afe1865c920fdd8f9a40ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4642491ddc86effe1ad13f8d701aaf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_054249f0a2afe1865c920fdd8f9a40ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d024fef5198055fa795d887b1ff97a22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_faf08cc6b82be29b7ef74e80c57a65b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d024fef5198055fa795d887b1ff97a22
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d205d824744c99f4c35cab1f3f941eaf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c50dea1dda32d4a52e3581e32ca51fd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d205d824744c99f4c35cab1f3f941eaf
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bb6cbc3acae11a0b9f7030af9140fdcf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c87bb44478def7030f1d0fa013bb6be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb6cbc3acae11a0b9f7030af9140fdcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61455d2f61795a89d6a46b141cc097bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d024fef5198055fa795d887b1ff97a22
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5d2530866212a1b84361e371bb9dda6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 19, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd9daeb3eb19de76ec7c9824b479a8ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5d2530866212a1b84361e371bb9dda6
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8056ef70b170b529e80e646d27dcbfca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2679ba5ca780504532ec7f310a746da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85d5dfa7e1583715dcb9d07cd7c23bc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 21, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48a3adf12498de01fe777a99c1a487b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85d5dfa7e1583715dcb9d07cd7c23bc4
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_360fce3cf3a1a4b19de16a6d90d211c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c62882ad3b335cf9686ff59981f62a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8a180f544f8ec879c0b891a71c54cdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c62882ad3b335cf9686ff59981f62a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc8e852a8b169701196061d2f57625b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40811518f5a9d14bf92996cd0fa0a985(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc8e852a8b169701196061d2f57625b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9455ada8b3df141dff0e9265f56fe272(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea6bd8a9d2776a3b0a7eb3b3e5113dbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_649b048704d99175d8d52787524b27ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec275da2578d87f9f2d7ad00f78836b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_649b048704d99175d8d52787524b27ef
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e073259f04ca216b5f0b5f372adda8d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d205d824744c99f4c35cab1f3f941eaf
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aff20348831a66eae026399691c32693(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a45722b01cab8b98414906681a6d6598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20348831a66eae026399691c32693
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faf08cc6b82be29b7ef74e80c57a65b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d024fef5198055fa795d887b1ff97a22
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eff0324f2819fa8b3c2c0c2f4aa9807(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f4c55aa955bf34a7efb9065464a29fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae6cf9a7e3386c35d6edc5988ff88a02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09e8d40ae31d91fe3cc0429cfd12e3af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6cf9a7e3386c35d6edc5988ff88a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea5e7f3b30c512f78f9c05117aea3f2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b47c69da95951aafd90cc483cb8aa57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea5e7f3b30c512f78f9c05117aea3f2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7bf6645e8e77bad145cc634528b3ab5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9922a0afbccc1764562119d2004f6b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bf6645e8e77bad145cc634528b3ab5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82de7d06ad222e4c3777476fe236b060(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94288d76a179eb0e2819cfa075583743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82de7d06ad222e4c3777476fe236b060
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1581808eb6ec979a767aad6efc3a4aa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20348831a66eae026399691c32693
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4cd8c8633a16ec2919a0ef6784469fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cec0bd9c5a4adcc077b31f0f62c99f42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cd8c8633a16ec2919a0ef6784469fa
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0cb40190bec187b1ed63fa462133b77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_058fde30dd6c62dd3adf6eefb625d322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0cb40190bec187b1ed63fa462133b77
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61455d2f61795a89d6a46b141cc097bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d024fef5198055fa795d887b1ff97a22
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac55277edbfd1a6f590a306f9dbd2704(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d649dade60a7eaab6e51e4194fc89df1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac55277edbfd1a6f590a306f9dbd2704
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32af85eabcbea1605231f5c483595b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_649b048704d99175d8d52787524b27ef
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6e6148db28e0e3dde820640420630a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8abef18fd2d80ad19f90e3ed65fbaf5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b529c704aa3a6e5681edc1da8d0bb829(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8abef18fd2d80ad19f90e3ed65fbaf5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94288d76a179eb0e2819cfa075583743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82de7d06ad222e4c3777476fe236b060
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0fa79dfa77797e2e5ccb9407d6ec65e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4168ad9cec8b9ed7a9a1d44ccaf7eef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bf6645e8e77bad145cc634528b3ab5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23e2b9b35f8a5ef525814c67fc18f03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c62882ad3b335cf9686ff59981f62a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_711566cf3b55b14a738300b72755df97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_045e71e8bf009aca4be2a396c06d1f41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_711566cf3b55b14a738300b72755df97
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1581808eb6ec979a767aad6efc3a4aa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20348831a66eae026399691c32693
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5427ab4683ca2a4866a85052020bfcbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51ff0e8ee62d2740f085826fb40cfe6c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8805ec4d37e6d3a4ce6b060b1b44c51c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ff0e8ee62d2740f085826fb40cfe6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ce1f2b44b4762face2ffcdf7e7bbb7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c99759386d3e4ae8a47c051f8492878(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ce1f2b44b4762face2ffcdf7e7bbb7e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c99759386d3e4ae8a47c051f8492878(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ce1f2b44b4762face2ffcdf7e7bbb7e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45722b01cab8b98414906681a6d6598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20348831a66eae026399691c32693
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40811518f5a9d14bf92996cd0fa0a985(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc8e852a8b169701196061d2f57625b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b529c704aa3a6e5681edc1da8d0bb829(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8abef18fd2d80ad19f90e3ed65fbaf5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91f416c4191796c4d5135e34a46c6567(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cd8c8633a16ec2919a0ef6784469fa
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d649dade60a7eaab6e51e4194fc89df1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac55277edbfd1a6f590a306f9dbd2704
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_045e71e8bf009aca4be2a396c06d1f41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_711566cf3b55b14a738300b72755df97
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33fd98d51626e012f0e6c53d94a1ab8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8056ef70b170b529e80e646d27dcbfca
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a57b01e50d523e3be6f0f64e07c01f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20348831a66eae026399691c32693
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_183cce3ce3f64eb6ca042bffafb744d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea5e7f3b30c512f78f9c05117aea3f2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_77d5db6169bfc60b75771a7bed209ac4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c8e4226d8f1fc538a12de9726d3d13b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77d5db6169bfc60b75771a7bed209ac4
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f2bd365c50097d7b1a39c08834997c2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f90620aa7df2dbd7254c49b7fc0917ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2bd365c50097d7b1a39c08834997c2f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf76680409a4df1f29cd4f4d76f0d947(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c62882ad3b335cf9686ff59981f62a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_058fde30dd6c62dd3adf6eefb625d322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0cb40190bec187b1ed63fa462133b77
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_44a916e93e4675f31e923e78d35325a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d095bc9385a3df297f75c7067ca4060(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44a916e93e4675f31e923e78d35325a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e9152c6dc8c243919559052c758e49d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff20348831a66eae026399691c32693
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4642491ddc86effe1ad13f8d701aaf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_054249f0a2afe1865c920fdd8f9a40ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e76066d9a2d6946c414473609c872b24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9440bb27cd96bb24f748c7bfc2d9666c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e76066d9a2d6946c414473609c872b24
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38df51963a17e3e3a45ba5d650898e86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df40cd3de31bded0aac7e453a568e906(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38df51963a17e3e3a45ba5d650898e86
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b9e810e343c487ac234c650e829a820(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 4, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89fe635eb41ccaa56fe0ddf8efbba392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b9e810e343c487ac234c650e829a820
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_20591f410e52f27dabcbe8bb2d1a6823(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3f3649d31615c48f84f43ebd2b30cf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20591f410e52f27dabcbe8bb2d1a6823
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b07572d258ceb91f1eae02ff1703e8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 19, 32768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7d9a5d802adc5205d65dd0ccc4c876e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b07572d258ceb91f1eae02ff1703e8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc26444416e0d7a879126e040f3498dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53bc75a32e4e7d0e2e2404a847bf90e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc26444416e0d7a879126e040f3498dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae340f9d767831dfba5771ca782bbc82(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21, 16384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_748a997320b22922ba1b69e25dce98f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae340f9d767831dfba5771ca782bbc82
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_24bd8fdfc32f465ad5857048d7bfb0d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b017d12b2bad3b48f7fab079f0217e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24bd8fdfc32f465ad5857048d7bfb0d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98b0906702cedd93b24008fb57f79973(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee2fdf6a47abd94cbd001146538d43cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98b0906702cedd93b24008fb57f79973
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fa36ed2eb840f1a3e2c1658a3939574(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03515881b92a102fb1b83ea0a91d17c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa36ed2eb840f1a3e2c1658a3939574
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac0718b96bd3a7cd6815d1506a48b1bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28f6e0e7174399dcdd6ab949085a3b08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac0718b96bd3a7cd6815d1506a48b1bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ec30abf6f0e2387d9a2264135a45175(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39d04f2e65c6d26fddf5589d7119b293(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ec30abf6f0e2387d9a2264135a45175
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ed504cb4f019ff58fad5199c32b1b12(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5b43f08b90d380bc112edf81c27a49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ed504cb4f019ff58fad5199c32b1b12
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_668f2eb8a83960a8ae7cc851bef86aca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c5ad1d7d9a0e621a506b843c8749fef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_668f2eb8a83960a8ae7cc851bef86aca
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9a10c72ccdee220764146775b8de339(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3cead83cbdea9fc3d912ca8c040924f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9a10c72ccdee220764146775b8de339
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9440bb27cd96bb24f748c7bfc2d9666c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e76066d9a2d6946c414473609c872b24
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_021029c97e2600e702c0e70cb0fd4888(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a866075fcefd99dcddaa769e0518713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_021029c97e2600e702c0e70cb0fd4888
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8d40418e58f9cd2145d27d87a2e8cf50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a4b333ae743e72997a975fa4f6912bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d40418e58f9cd2145d27d87a2e8cf50
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d984306bd0af8c55a735492306d614d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51483bc5e7aa01e70a558a5c3e38a2e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d984306bd0af8c55a735492306d614d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4cfbb7dc4a72e6cbb7de39779801b025(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2aaba52dbb59bec0cb1736c14707002c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cfbb7dc4a72e6cbb7de39779801b025
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce3a28a11830f283a6249483385c17cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 4096], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0968ff6043495a211b8c203e596d03fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce3a28a11830f283a6249483385c17cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f2bb76c02bc493cef6a8a41fc60148b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_952e5bf6e667c65888d524ecf9b84322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2bb76c02bc493cef6a8a41fc60148b0
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_367636658b5de91dd5f97b4eacf4a1d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_586c4c31691275656f7187b4d180bc32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_367636658b5de91dd5f97b4eacf4a1d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e741ab8d84f46f6b791a8f6b71e6a501(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_525d502321b12046ba13f247e15401ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e741ab8d84f46f6b791a8f6b71e6a501
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c889d30ed88fb18c2fe81dafcba1ab30(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8151f8de34f6e46eeb15398c36c07dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c889d30ed88fb18c2fe81dafcba1ab30
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f3649d31615c48f84f43ebd2b30cf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20591f410e52f27dabcbe8bb2d1a6823
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea5fc42f590e012eabac5a476f255d15(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2686a9973463f5b844716fd00f262104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea5fc42f590e012eabac5a476f255d15
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e0e7d1b329175b284f71a8abfb81abf6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba026f389355362e45032e380647b9f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0e7d1b329175b284f71a8abfb81abf6
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55402a5b13a72833eaab0052e3fd8c46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00cc10e31147c7f7346cb4a257c8d9fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55402a5b13a72833eaab0052e3fd8c46
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8a161ad596a97541f9643176bb0e4ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f09a3704a9f2535d8a390acc1adccc0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a161ad596a97541f9643176bb0e4ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_952e5bf6e667c65888d524ecf9b84322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2bb76c02bc493cef6a8a41fc60148b0
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_656a301c728119e985b7f5399867d781(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5237bb41ba7a938a9a88310fac78a712(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_656a301c728119e985b7f5399867d781
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a2e1aea82226d8af9006cae5202afd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 8192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f427cc29ac97f5e6d001c3a8c67107ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a2e1aea82226d8af9006cae5202afd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba4ddb72438ce82f5375a39d878b48bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6c13eb6cfb2281a76b6e44b450afc2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba4ddb72438ce82f5375a39d878b48bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e06ba60616aed2e9ad14fd142d907d01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c426e8cb061e5eeb8a129bf61d3c937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e06ba60616aed2e9ad14fd142d907d01
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586c4c31691275656f7187b4d180bc32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_367636658b5de91dd5f97b4eacf4a1d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f48bcc8cb40e627d8f4a772b8b46659(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47a8ee6ec078ae67571595476b859651(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f48bcc8cb40e627d8f4a772b8b46659
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b059d53f56b72890258ab29915734cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f301f27114d90e2e2a959fd6f7d0de3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b059d53f56b72890258ab29915734cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b7c7cf968244f0828bf26d9873df49e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f114aefac104e24f24ec88557a268fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b7c7cf968244f0828bf26d9873df49e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f114aefac104e24f24ec88557a268fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b7c7cf968244f0828bf26d9873df49e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cead83cbdea9fc3d912ca8c040924f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9a10c72ccdee220764146775b8de339
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03515881b92a102fb1b83ea0a91d17c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa36ed2eb840f1a3e2c1658a3939574
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f09a3704a9f2535d8a390acc1adccc0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a161ad596a97541f9643176bb0e4ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7e4d120467e9e35e1954f15e9e468b3b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13271e28c0258142d1e78bf2d2d45b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e4d120467e9e35e1954f15e9e468b3b
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2686a9973463f5b844716fd00f262104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea5fc42f590e012eabac5a476f255d15
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c426e8cb061e5eeb8a129bf61d3c937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e06ba60616aed2e9ad14fd142d907d01
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e65689b4ed9607742841b0616744f484(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1e0778812c411b1b83f7524729354ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e65689b4ed9607742841b0616744f484
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eedff497289ee8ab4d6f3a404a9f0712(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcbd3618dd3ff8fcdabdf4eaa4303954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eedff497289ee8ab4d6f3a404a9f0712
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c74d8c80be2952664acacb6f2de138f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75b6ec2435a89988260f0f2c06a29920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c74d8c80be2952664acacb6f2de138f5
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_64dd857047f48d819303bb5b09f8174e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3754c70218c0d2cb42114f7b59c0add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64dd857047f48d819303bb5b09f8174e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b25d56832ad1a0894f72d6c7cfe4f746(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6dfe5c53d03211156a74fc7d005c83b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b25d56832ad1a0894f72d6c7cfe4f746
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bad35c2e6072b165cb8508d31b560439(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70527a76b6a4e5e7b77e504fc3efb44a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bad35c2e6072b165cb8508d31b560439
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8151f8de34f6e46eeb15398c36c07dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c889d30ed88fb18c2fe81dafcba1ab30
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2161a46fda845d18efef8c14226f00e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75962601e28b9e2de346a53cc830210b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2161a46fda845d18efef8c14226f00e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0fe9bb80032f0a94fd839854f4a2da54(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d155e2217d29a45d712763541e6affa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fe9bb80032f0a94fd839854f4a2da54
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5621015be0b6c5a3adc170b73964b548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0de353aa631762811c89a0917c342412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ac466bc200c525c33d05c40d748fd2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e29479ac38d134020a295706d6692f55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adb424fe3f0f45c8be0cbc1a1236525f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_05defb2896453c0eeefb851c6d49b524(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_383112ebd1127872d5b784cb67e71865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05defb2896453c0eeefb851c6d49b524
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e9029d46c448da0e787b8654c159711(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f4be2e7a1598b14a05bc62cde97981a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05defb2896453c0eeefb851c6d49b524
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19b2098716b3f61726d925b839ec06ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb6cc1d942ab1ae6fa30b2578f3256eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7826ebb1f33b374612d385f1df2fd6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab4c7d15246680000ee51613d599bed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83c6a3df30eafb50a14ed4a07c81a48a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ca83bc5944e19273c07d0ae980abdcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_574c9af912b2827a5c0cac1b9401979f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f12b932e945991664bff88a31fd462bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0de353aa631762811c89a0917c342412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d249f8138bd11f0dd969f6508d904bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d4ff79b3664dc227d971719c509d1e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa029ebd5ddb402ecea81b928d86ed3f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfde49e092f2eeec437c34d866ce6119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa029ebd5ddb402ecea81b928d86ed3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cab51d185504a5f9e2dc1ae8f4608bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9922a0afbccc1764562119d2004f6b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bf6645e8e77bad145cc634528b3ab5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e5415a21fb8d3deb1ffd1faa35cc8aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3d24fb59baa97f591291940b74c4190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3852ed4f35e9656e6ff0ba1e91f00b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9688ed7a46b1f0085cee9b6607e3790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adb424fe3f0f45c8be0cbc1a1236525f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa064b246a718863764a23d8d625f519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d987e3246647a2f724ea52f2e459606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8166c0cfc309b2387ebf1db8ef6ed6e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc06b0427d29ae5a3a0da37009eb034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e5415a21fb8d3deb1ffd1faa35cc8aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d732a622b13217bc83187264c46b4821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4168ad9cec8b9ed7a9a1d44ccaf7eef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bf6645e8e77bad145cc634528b3ab5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6628def8cbd5df5f54c5eb6af0426368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_903f4b98e49a96512c7a7731c0a66f05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bf6645e8e77bad145cc634528b3ab5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3d24fb59baa97f591291940b74c4190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96f58cc82cb111eaf2d2a27e7dd4edf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46fb4a70b4d773a4470e49af4bb31384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa029ebd5ddb402ecea81b928d86ed3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b521739fd761709f9c619e295e65f454(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b521739fd761709f9c619e295e65f454(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f12b932e945991664bff88a31fd462bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7826ebb1f33b374612d385f1df2fd6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc06b0427d29ae5a3a0da37009eb034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b6a317f00963a84ca597db1f011ba5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa064b246a718863764a23d8d625f519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_903f4b98e49a96512c7a7731c0a66f05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bf6645e8e77bad145cc634528b3ab5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4382997c186728c8f4f8324424537d9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_235541be9401b3ede9bcfd667d4672d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4c9a73ace2dc83a8daba5c9b35d5fb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8474f6edec6f089761383630808911b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa029ebd5ddb402ecea81b928d86ed3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6ee68b136b50696e330b6316b6db838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa029ebd5ddb402ecea81b928d86ed3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0569eda90e5c51c4d35c4cd8adcc606b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9688ed7a46b1f0085cee9b6607e3790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5de8018010fe175bc6973ac1b469262d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa029ebd5ddb402ecea81b928d86ed3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ed2c334ec8da8bc02810cad40cfb2f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050b2f18ee236ae8a711c4340aac3e8f
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()