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
    class PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52936884cece313096bd0823c46f5308(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9100485444068909]]], [[[1.2505614757537842]]], [[[1.207377314567566]]], [[[1.2603110074996948]]], [[[1.8400535583496094]]], [[[1.1723406314849854]]], [[[1.7790600061416626]]], [[[1.7147533893585205]]], [[[1.519740343093872]]], [[[0.9402708411216736]]], [[[1.3458735942840576]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f6b2b75d5617786c3bf5b1a7044d0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92362dc76d7126a7a9d5f7fcc09bfcd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.813521385192871]]], [[[1.7184243202209473]]], [[[1.2319406270980835]]], [[[1.0926905870437622]]], [[[1.086316466331482]]], [[[1.1540483236312866]]], [[[1.3221023082733154]]], [[[1.4302207231521606]]], [[[1.6104097366333008]]], [[[1.786789894104004]]], [[[1.1484756469726562]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c578caf52fe5fb7096e327918c18ca5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8158645629882812]]], [[[1.61286199092865]]], [[[1.2113714218139648]]], [[[1.0517899990081787]]], [[[1.9453657865524292]]], [[[1.1764276027679443]]], [[[1.599045753479004]]], [[[1.9410607814788818]]], [[[1.2618427276611328]]], [[[1.00847327709198]]], [[[1.6717091798782349]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d966500c14b05f3ce2c9fd48253cef65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00d0ae7c296c5a3ad0a716fe26d8ccc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84e16270d1e68d56964321be8e11a4b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5097a49427db0051b213fa6aba984436(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4889c36f59f959b6a8c25f813c8e8188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.0315189361572266]]], [[[1.4978537559509277]]], [[[1.351078748703003]]], [[[1.5752840042114258]]], [[[1.558007001876831]]], [[[1.140337347984314]]], [[[1.9190685749053955]]], [[[1.6311914920806885]]], [[[1.1684446334838867]]], [[[1.169787883758545]]], [[[1.560655117034912]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_440bd35d092adce36130b5f86c9e9641(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70289013693d5f124a1b7af414a9f97d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d690746ed0538359db72fcb93dc68b91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6f3048e88ac6150ed0328f6a94eefed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c4ec5fa2f20f98ed4535f045ee44ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a09d109bc045a8d5e14adbeb1e2185f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7057b66ca527faa521f2734dc52222b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec6238008661aadc4be28d55f5ebc3be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_363d77e64be77423d65a120d877c4214(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_418ea408d1f3cfed87d2e23e70db2028(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.6414575576782227]]], [[[1.4313457012176514]]], [[[1.2378675937652588]]], [[[0.9781988859176636]]], [[[1.419010043144226]]], [[[1.181267261505127]]], [[[0.9421247839927673]]], [[[1.7694975137710571]]], [[[1.7152119874954224]]], [[[1.2278752326965332]]], [[[1.866539716720581]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7112c5bba9b0fd64df3c488ce100069c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c4c1e551bf0e82a052e66ead97c0239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c3a5f6e61df906275e6500e973b3194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8aefe0130da35bfffe05ac43b792b161(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_049d2739b2cfa446c164f97b88b26cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9100485444068909]]], [[[1.2505614757537842]]], [[[1.207377314567566]]], [[[1.2603110074996948]]], [[[1.8400535583496094]]], [[[1.1723406314849854]]], [[[1.7790600061416626]]], [[[1.7147533893585205]]], [[[1.519740343093872]]], [[[0.9402708411216736]]], [[[1.3458735942840576]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f1cfad677b86cf950cb3464718eb474(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38e300e74087aebb17eb9168db7fe9f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f1cfad677b86cf950cb3464718eb474
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2eb6b8856569dbe38e2b7c8e803c37c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.813521385192871]]], [[[1.7184243202209473]]], [[[1.2319406270980835]]], [[[1.0926905870437622]]], [[[1.086316466331482]]], [[[1.1540483236312866]]], [[[1.3221023082733154]]], [[[1.4302207231521606]]], [[[1.6104097366333008]]], [[[1.786789894104004]]], [[[1.1484756469726562]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_60a26869e7cfafb048e683b044063129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8158645629882812]]], [[[1.61286199092865]]], [[[1.2113714218139648]]], [[[1.0517899990081787]]], [[[1.9453657865524292]]], [[[1.1764276027679443]]], [[[1.599045753479004]]], [[[1.9410607814788818]]], [[[1.2618427276611328]]], [[[1.00847327709198]]], [[[1.6717091798782349]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c1ec7cfef6c493afc7d7bc6ecf2ffde(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_edd8afafb1391af4429346028b1e3f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c1ec7cfef6c493afc7d7bc6ecf2ffde
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5dd23500a49059f20d6ffe7a9e1fe1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0be0c250ea79774a5be8d5a3ad1833bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5dd23500a49059f20d6ffe7a9e1fe1e
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b636ab67a897ea2db83993ceed91aa68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f18392f059664a674186b2ef8de94c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b636ab67a897ea2db83993ceed91aa68
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a408293fdf685dcac77f0d7cb675d40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b716b3da7c9bf141a121d23a984ae90f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a408293fdf685dcac77f0d7cb675d40
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45425d44f989d090c732248ff048b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.0315189361572266]]], [[[1.4978537559509277]]], [[[1.351078748703003]]], [[[1.5752840042114258]]], [[[1.558007001876831]]], [[[1.140337347984314]]], [[[1.9190685749053955]]], [[[1.6311914920806885]]], [[[1.1684446334838867]]], [[[1.169787883758545]]], [[[1.560655117034912]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_1d3e8aa7b7d5f94bb47b16c6e3969a38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6b51f06cacb4cf2a6df95ffd699a60c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d3e8aa7b7d5f94bb47b16c6e3969a38
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a1abad084cd2eab75530abad6a3a7f93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a888dbeb18df74417a0cb58732e4216d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1abad084cd2eab75530abad6a3a7f93
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6689afac2505b798ca69ba2261c5c0a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b2601dd00bee5621ec34e50dc9a72f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6689afac2505b798ca69ba2261c5c0a3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a27b52e3345330ea952c4b0f2eb5113(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae07672c44d2a365ff95ad3417e2ab66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a27b52e3345330ea952c4b0f2eb5113
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac441acba2895514a61475c5c4a0443d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fc9d99ff2150e49aa7681aad923334f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac441acba2895514a61475c5c4a0443d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1375cd43df8dd3ea80117cb6179a8c59(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_494307d466e90fc3f03b6a6790588f9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1375cd43df8dd3ea80117cb6179a8c59
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_978a3645856e0c2c27ff801f9921170d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_844d2ba3711ab39f5459c94978be95ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_978a3645856e0c2c27ff801f9921170d
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c16c14210ae2f6bc95f87fdf19ec4840(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe87bf8fddafc532aecfb535af334e77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c16c14210ae2f6bc95f87fdf19ec4840
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8bcb27f9bc1c8f303de56483ebfcb601(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9b8d17fe588421ceb0227be9f8a9fe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bcb27f9bc1c8f303de56483ebfcb601
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3ba2ba7a8d18105a1cdc777bbcdaaed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.6414575576782227]]], [[[1.4313457012176514]]], [[[1.2378675937652588]]], [[[0.9781988859176636]]], [[[1.419010043144226]]], [[[1.181267261505127]]], [[[0.9421247839927673]]], [[[1.7694975137710571]]], [[[1.7152119874954224]]], [[[1.2278752326965332]]], [[[1.866539716720581]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c8a9f4ac37c2a1edcec322251de16df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09073e368e0b92f6c802ff2b61d2fdf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c8a9f4ac37c2a1edcec322251de16df
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2fc76c9d90bee4901bddd7a45d3b6985(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_336f0b22bc7ae33f505facfc7255076b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc76c9d90bee4901bddd7a45d3b6985
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_410c0179ce327325eda68e88526e94da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8b81b39f6f5a60cd11f08a1724b4352(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_410c0179ce327325eda68e88526e94da
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c82546ded332f4b321abcfb04df3d45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9100485444068909]]], [[[1.2505614757537842]]], [[[1.207377314567566]]], [[[1.2603110074996948]]], [[[1.8400535583496094]]], [[[1.1723406314849854]]], [[[1.7790600061416626]]], [[[1.7147533893585205]]], [[[1.519740343093872]]], [[[0.9402708411216736]]], [[[1.3458735942840576]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0873feb5994211575fc19ce8ae9aef1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e45e38e89988b4f33e6feca8281a82e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.813521385192871]]], [[[1.7184243202209473]]], [[[1.2319406270980835]]], [[[1.0926905870437622]]], [[[1.086316466331482]]], [[[1.1540483236312866]]], [[[1.3221023082733154]]], [[[1.4302207231521606]]], [[[1.6104097366333008]]], [[[1.786789894104004]]], [[[1.1484756469726562]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_4c202e69d28ca7d122dd833ce806288d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8158645629882812]]], [[[1.61286199092865]]], [[[1.2113714218139648]]], [[[1.0517899990081787]]], [[[1.9453657865524292]]], [[[1.1764276027679443]]], [[[1.599045753479004]]], [[[1.9410607814788818]]], [[[1.2618427276611328]]], [[[1.00847327709198]]], [[[1.6717091798782349]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0e5476d5843d2d3c7a086fd91edbde3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28b24df090ebcd9026ea714ce36b92cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d01d56b211862d4e69bfe9905dad52c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0e5ca8bd9cec4e1b18eb4ecf20dcac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43e0ef4db278b0ca4ed597741a17380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.0315189361572266]]], [[[1.4978537559509277]]], [[[1.351078748703003]]], [[[1.5752840042114258]]], [[[1.558007001876831]]], [[[1.140337347984314]]], [[[1.9190685749053955]]], [[[1.6311914920806885]]], [[[1.1684446334838867]]], [[[1.169787883758545]]], [[[1.560655117034912]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_95df247a3a5b1005ab63b233ae3efa1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2764a7d4b695a44241591e569f786808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ced9e0776d04c69066bf96a5fa55bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63ec144ba4bf801f91fa8a06f03a2494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc9ace5be76dbb1305bc338b5eea46d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f9d5a4a234a79abe034b91064346369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4eccfb8f7b8a0978de92a90197ff225d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c4cef2f12f08b920c8e295fbd544d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b6dfbc4b92eab05043577be50542d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e63b75a387fd593429df015dc178be6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.6414575576782227]]], [[[1.4313457012176514]]], [[[1.2378675937652588]]], [[[0.9781988859176636]]], [[[1.419010043144226]]], [[[1.181267261505127]]], [[[0.9421247839927673]]], [[[1.7694975137710571]]], [[[1.7152119874954224]]], [[[1.2278752326965332]]], [[[1.866539716720581]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b188fd89f2adbae2e9984912c6e6b4c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cadc06be8f1eb6a1c8f0a887553c58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc319f33e996de4ab6fe2eff0b0e312d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()