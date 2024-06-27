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


    class TestPrimitiveOp_027de25ca62ecfe589da6da7a7f8b9f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.533012866973877]]], [[[1.4788471460342407]]], [[[1.489168643951416]]], [[[1.0151605606079102]]], [[[0.9045605659484863]]], [[[1.4528697729110718]]], [[[1.3621773719787598]]], [[[0.8800214529037476]]], [[[1.098995566368103]]], [[[0.9575161933898926]]], [[[0.8935641050338745]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_6978449df9e3a96d46a3d917c11ed029(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52f40fd6ca40414470d461a349b5486a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9670681953430176]]], [[[1.3227744102478027]]], [[[1.5465021133422852]]], [[[1.2401866912841797]]], [[[1.6133918762207031]]], [[[1.8724260330200195]]], [[[1.2263115644454956]]], [[[1.6836156845092773]]], [[[1.2098950147628784]]], [[[1.4763588905334473]]], [[[0.94552081823349]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_a872638b13c7edab5e48aeeffac78e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5707824230194092]]], [[[1.7985334396362305]]], [[[1.0667887926101685]]], [[[1.4934873580932617]]], [[[1.4714419841766357]]], [[[0.9734923243522644]]], [[[0.9932284355163574]]], [[[1.672088623046875]]], [[[1.2482717037200928]]], [[[1.6366945505142212]]], [[[1.5233125686645508]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_e99557c19d5dd94a9a8945918c07598b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8d9f331d2e346bc563773742dc822e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bd57e2ae55edcca533d0e7906762ce9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18ce098baeade5dd45c51e2cff3fd0e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.9437956809997559]]], [[[1.6161741018295288]]], [[[1.1814286708831787]]], [[[1.0405232906341553]]], [[[1.1228320598602295]]], [[[1.0236417055130005]]], [[[1.5329616069793701]]], [[[1.0212854146957397]]], [[[1.1669082641601562]]], [[[1.4491333961486816]]], [[[0.9972606897354126]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c54d84a910707c8b57d264c5b4b0a9f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70289013693d5f124a1b7af414a9f97d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f999ef180efbd374d90bbc6b4fb30a7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5062e08f8801bedd1abc6a0ee0b44976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c4ec5fa2f20f98ed4535f045ee44ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd66bee1957c52d59d070eaaf163d58a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0138382c583d2989201438858e8ed232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be99740411079c9fc3064dc9a83dc94d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_363d77e64be77423d65a120d877c4214(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30dd3f5e926f569e324c39e4e4702aa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4122763872146606]]], [[[1.0732494592666626]]], [[[1.7405369281768799]]], [[[1.4431321620941162]]], [[[1.737332820892334]]], [[[1.108514428138733]]], [[[1.7897511720657349]]], [[[1.4065593481063843]]], [[[1.2843037843704224]]], [[[1.3246482610702515]]], [[[1.3931899070739746]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e96ddb82bd005bad544f7ed568227ec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c4c1e551bf0e82a052e66ead97c0239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bf9b441d71e46b3d8fa404e199f9e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6caa0c8fce7fddddad5e78847a94d067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.533012866973877]]], [[[1.4788471460342407]]], [[[1.489168643951416]]], [[[1.0151605606079102]]], [[[0.9045605659484863]]], [[[1.4528697729110718]]], [[[1.3621773719787598]]], [[[0.8800214529037476]]], [[[1.098995566368103]]], [[[0.9575161933898926]]], [[[0.8935641050338745]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    
    class PrimitiveOp_536ce0b6f0b4b76a267fd02125327cfd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24ed42c96ed9031942f355905a3fb116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_536ce0b6f0b4b76a267fd02125327cfd
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64fb3fbfeeec3380c13b9afcae2ecf6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9670681953430176]]], [[[1.3227744102478027]]], [[[1.5465021133422852]]], [[[1.2401866912841797]]], [[[1.6133918762207031]]], [[[1.8724260330200195]]], [[[1.2263115644454956]]], [[[1.6836156845092773]]], [[[1.2098950147628784]]], [[[1.4763588905334473]]], [[[0.94552081823349]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_1b065dc8a68ecc62a7d47c978fe804a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5707824230194092]]], [[[1.7985334396362305]]], [[[1.0667887926101685]]], [[[1.4934873580932617]]], [[[1.4714419841766357]]], [[[0.9734923243522644]]], [[[0.9932284355163574]]], [[[1.672088623046875]]], [[[1.2482717037200928]]], [[[1.6366945505142212]]], [[[1.5233125686645508]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    
    class PrimitiveOp_a4d48a5db09e3fbca81e3e76d0e88978(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7da9643816064597889198611922d7b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4d48a5db09e3fbca81e3e76d0e88978
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57e5fa7dd9329fe40fc1ad8ff8f96151(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_852181ea5649c9180496a34c0841321d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57e5fa7dd9329fe40fc1ad8ff8f96151
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3f429dd5676265f688b245bcf3672dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b32da277ec5e3c69e10d145be3e19914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f429dd5676265f688b245bcf3672dd
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3731179e8eaada4ca6a1674fba4cc33c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.9437956809997559]]], [[[1.6161741018295288]]], [[[1.1814286708831787]]], [[[1.0405232906341553]]], [[[1.1228320598602295]]], [[[1.0236417055130005]]], [[[1.5329616069793701]]], [[[1.0212854146957397]]], [[[1.1669082641601562]]], [[[1.4491333961486816]]], [[[0.9972606897354126]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_3a2069259cea24b2269868cb6ad87348(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07241c0ba8a7989c32a0c48ec98ebef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a2069259cea24b2269868cb6ad87348
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_cc95533d4a556429e4ad51c081aeac9c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0214a7f622320138872642e3d83aae2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc95533d4a556429e4ad51c081aeac9c
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_18bb76cd136ad8fa06b491077e9676a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_090a02713e5cd3cad9d2d5a0f4e8b832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18bb76cd136ad8fa06b491077e9676a4
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_595ee9343d27b24e559c82f31d019d89(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aef3fa71a2944d6e9b197aa3b885eeb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_595ee9343d27b24e559c82f31d019d89
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e5c2223881101f4c552b8626232ac23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_187bc6604c0a7449a6d9a717a4aef337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e5c2223881101f4c552b8626232ac23
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_517a02560d03899f0e3727d67501f2e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_593a93dbc5e9cc6cba7cfb45aeb790df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_517a02560d03899f0e3727d67501f2e0
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_eb4b8ec419ba6260f46bc6d3d3e441ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4122763872146606]]], [[[1.0732494592666626]]], [[[1.7405369281768799]]], [[[1.4431321620941162]]], [[[1.737332820892334]]], [[[1.108514428138733]]], [[[1.7897511720657349]]], [[[1.4065593481063843]]], [[[1.2843037843704224]]], [[[1.3246482610702515]]], [[[1.3931899070739746]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48e7e145d352e75dc76732903915bfa6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e0259675817bb7368eb69309d5e52c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48e7e145d352e75dc76732903915bfa6
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_f342985309bd54621947b72b3d80211b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_baa1909add2758c81785cdfc0abbf3fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f342985309bd54621947b72b3d80211b
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2060adf114eeaa6629431aefb324c64b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.533012866973877]]], [[[1.4788471460342407]]], [[[1.489168643951416]]], [[[1.0151605606079102]]], [[[0.9045605659484863]]], [[[1.4528697729110718]]], [[[1.3621773719787598]]], [[[0.8800214529037476]]], [[[1.098995566368103]]], [[[0.9575161933898926]]], [[[0.8935641050338745]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_dbc7501f922691a6820de50a13a5016f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db5b1f35fdda3e33ee164f59b79bf7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.9670681953430176]]], [[[1.3227744102478027]]], [[[1.5465021133422852]]], [[[1.2401866912841797]]], [[[1.6133918762207031]]], [[[1.8724260330200195]]], [[[1.2263115644454956]]], [[[1.6836156845092773]]], [[[1.2098950147628784]]], [[[1.4763588905334473]]], [[[0.94552081823349]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_29a5b8492b7ff307883caa168d1cfa0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.5707824230194092]]], [[[1.7985334396362305]]], [[[1.0667887926101685]]], [[[1.4934873580932617]]], [[[1.4714419841766357]]], [[[0.9734923243522644]]], [[[0.9932284355163574]]], [[[1.672088623046875]]], [[[1.2482717037200928]]], [[[1.6366945505142212]]], [[[1.5233125686645508]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_b31788950695a631e466c42ed0490e4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b48b1f17b2cc6bc56a74678b9783c2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c808206f2519cfb1a5cc2267be4657eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4c0c568c2bcb4c21824113cc9abc56a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.9437956809997559]]], [[[1.6161741018295288]]], [[[1.1814286708831787]]], [[[1.0405232906341553]]], [[[1.1228320598602295]]], [[[1.0236417055130005]]], [[[1.5329616069793701]]], [[[1.0212854146957397]]], [[[1.1669082641601562]]], [[[1.4491333961486816]]], [[[0.9972606897354126]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_e79fa891373a24e03ee7d6b28f043e71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2764a7d4b695a44241591e569f786808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecd853d85ebef74ac45510b284f26681(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ef012421d6fe750dc718aa754bf76f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc9ace5be76dbb1305bc338b5eea46d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ce8db16167342f7085ae536bf6f7297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1eedbbd46be72debe58d8cc6ad45b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fedae25ba9b15c5e99ad07bbb3a1a4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b6dfbc4b92eab05043577be50542d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c4fff47550bcea99607d786dbfe2bcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4122763872146606]]], [[[1.0732494592666626]]], [[[1.7405369281768799]]], [[[1.4431321620941162]]], [[[1.737332820892334]]], [[[1.108514428138733]]], [[[1.7897511720657349]]], [[[1.4065593481063843]]], [[[1.2843037843704224]]], [[[1.3246482610702515]]], [[[1.3931899070739746]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36c5ef4b405349302552e3d128585455(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cadc06be8f1eb6a1c8f0a887553c58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8eb1d2de0e1a13523268b22248decdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()