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


    class TestPrimitiveOp_1797c29f6adea6eb1e6ae5f8c385c5f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1214570999145508]]], [[[1.4196274280548096]]], [[[0.9153375029563904]]], [[[1.7990394830703735]]], [[[1.0699777603149414]]], [[[1.7659509181976318]]], [[[1.7617671489715576]]], [[[1.3789520263671875]]], [[[1.4761914014816284]]], [[[1.3960072994232178]]], [[[1.2863178253173828]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_115e3c9f88deb3d22c82872e78611cf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8f3474445b6626ccca3c41b64724a74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.625505805015564]]], [[[1.0636988878250122]]], [[[1.2986936569213867]]], [[[1.8430451154708862]]], [[[0.9542919993400574]]], [[[0.951836884021759]]], [[[1.1712501049041748]]], [[[0.97215336561203]]], [[[1.452519416809082]]], [[[0.9949560165405273]]], [[[1.1478893756866455]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c9dbbf5604823f93b2004d18e9259849(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3180955648422241]]], [[[1.8995996713638306]]], [[[1.1273280382156372]]], [[[1.849107265472412]]], [[[1.6026138067245483]]], [[[1.8701281547546387]]], [[[1.0979362726211548]]], [[[1.7622225284576416]]], [[[1.2591562271118164]]], [[[1.9011240005493164]]], [[[1.343113899230957]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_a29cbd186fb750faee3aad95368d490a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3483fcba2f5fd0a049e5023b7bfa33b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f75d2405df8927cd851c77dd52658c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e041ced7cc963fb717920a5fe90d84b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4986821413040161]]], [[[1.1676080226898193]]], [[[1.819826602935791]]], [[[1.2086670398712158]]], [[[1.7440937757492065]]], [[[1.4067095518112183]]], [[[1.08768630027771]]], [[[1.2487809658050537]]], [[[1.0397902727127075]]], [[[1.7107311487197876]]], [[[1.6483170986175537]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_1c078da2b764b7a5d20f355a88a64f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70289013693d5f124a1b7af414a9f97d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6e16e0856285f4826e9036aaaa255c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd129afd280f9158b06d7ecc3a87ce4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c4ec5fa2f20f98ed4535f045ee44ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0a1bd4036179ef19b2be41882ea0733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_306142c175e3c21d3e8b155d21f13a80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_376b841f4f9e925d9749e6118cc33a76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_363d77e64be77423d65a120d877c4214(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac5fb19049f95abbfe5247b28194051a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8052839040756226]]], [[[1.666274070739746]]], [[[0.96114581823349]]], [[[1.4701937437057495]]], [[[1.279260277748108]]], [[[1.2332408428192139]]], [[[1.875383734703064]]], [[[1.5806314945220947]]], [[[1.2589390277862549]]], [[[1.6112051010131836]]], [[[1.4541015625]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_55ff915db8f0ce7a68926b0ab147f289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32fe7be0eb53dd5783432438c07eb9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2d6c0dde3f687bc49b675b00291c5e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c4c1e551bf0e82a052e66ead97c0239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_747f3c1af28084e7f705a3c70bd0c8b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c88667fc2befee533f07c4f08e1321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2eabcdf340a9a8394b1b81fe28da341
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_3eb3d9c80847252f9c7457a21596aed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1214570999145508]]], [[[1.4196274280548096]]], [[[0.9153375029563904]]], [[[1.7990394830703735]]], [[[1.0699777603149414]]], [[[1.7659509181976318]]], [[[1.7617671489715576]]], [[[1.3789520263671875]]], [[[1.4761914014816284]]], [[[1.3960072994232178]]], [[[1.2863178253173828]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    
    class PrimitiveOp_4fc414f41dcc0591633e6e6a35c3fca1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b9522b7cf887729a37ee332052e22cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fc414f41dcc0591633e6e6a35c3fca1
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ac9f676bf849eb27e8f4f3f63d8f6a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.625505805015564]]], [[[1.0636988878250122]]], [[[1.2986936569213867]]], [[[1.8430451154708862]]], [[[0.9542919993400574]]], [[[0.951836884021759]]], [[[1.1712501049041748]]], [[[0.97215336561203]]], [[[1.452519416809082]]], [[[0.9949560165405273]]], [[[1.1478893756866455]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_74b812edcb47380e50d350f256075ea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3180955648422241]]], [[[1.8995996713638306]]], [[[1.1273280382156372]]], [[[1.849107265472412]]], [[[1.6026138067245483]]], [[[1.8701281547546387]]], [[[1.0979362726211548]]], [[[1.7622225284576416]]], [[[1.2591562271118164]]], [[[1.9011240005493164]]], [[[1.343113899230957]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    
    class PrimitiveOp_ce47149584ffdc8b6a38666fc22c474c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3293f9723bb0c136c05ac6cbe7335136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce47149584ffdc8b6a38666fc22c474c
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce1984237bbbf3a5e490c15ae64ad2c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eefbf487b1db070b472ee8703f020a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce1984237bbbf3a5e490c15ae64ad2c8
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0067f90b7ade1271bdabca7b644392bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9208f1eb26bb2a02283647b3fafbc04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0067f90b7ade1271bdabca7b644392bb
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_385090e04a25e895d65f30f9fb103287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4986821413040161]]], [[[1.1676080226898193]]], [[[1.819826602935791]]], [[[1.2086670398712158]]], [[[1.7440937757492065]]], [[[1.4067095518112183]]], [[[1.08768630027771]]], [[[1.2487809658050537]]], [[[1.0397902727127075]]], [[[1.7107311487197876]]], [[[1.6483170986175537]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_ac668c3a05a55aca151a9802867a1608(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d8df75d0c12db0c8a68fb03b4f7dec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac668c3a05a55aca151a9802867a1608
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_414851031e3cdfbec42040f30074ae21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d12016d5faba813ca8606d0fe945ad52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_414851031e3cdfbec42040f30074ae21
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc86d097329813f2fa4dd3011e3f7130(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fddc6946f51f740c72265d2f178fee79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc86d097329813f2fa4dd3011e3f7130
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_322c5eadfe1edebee1d6d85f640f5156(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c1002318fcba6e16e2a9a4d98a26a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_322c5eadfe1edebee1d6d85f640f5156
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_40f0557f39b9c8a55e9b7880c1fead42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa03ae1ca3b1207a21cf8efa7dc23984(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0557f39b9c8a55e9b7880c1fead42
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_56ab86bf3c439e270305c53c5398499a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4aef52f5e10615de3fd7227470199c87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56ab86bf3c439e270305c53c5398499a
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6d6de803cefe6d4e830922c2dd9fb180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aefe0130da35bfffe05ac43b792b161
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8052839040756226]]], [[[1.666274070739746]]], [[[0.96114581823349]]], [[[1.4701937437057495]]], [[[1.279260277748108]]], [[[1.2332408428192139]]], [[[1.875383734703064]]], [[[1.5806314945220947]]], [[[1.2589390277862549]]], [[[1.6112051010131836]]], [[[1.4541015625]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_48900e1240e75974f21f481f3acbe486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f4e1e09cd291a3cc39ccfd8cc3d879d
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9da30367cf51391adbaa3254a7cb6599(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9f1421358c532fbc067328c6921d37d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9da30367cf51391adbaa3254a7cb6599
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_91ab8c8d392b0617b54fdc35cd095f6e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ded7c903092531fbe5be14f9c8b9c124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91ab8c8d392b0617b54fdc35cd095f6e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c06231a677567bfd988436381e9495f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.1214570999145508]]], [[[1.4196274280548096]]], [[[0.9153375029563904]]], [[[1.7990394830703735]]], [[[1.0699777603149414]]], [[[1.7659509181976318]]], [[[1.7617671489715576]]], [[[1.3789520263671875]]], [[[1.4761914014816284]]], [[[1.3960072994232178]]], [[[1.2863178253173828]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_4d0f63e4faa05381dc34c935abe4db7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55699d27fd296aaad4535372355d92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.625505805015564]]], [[[1.0636988878250122]]], [[[1.2986936569213867]]], [[[1.8430451154708862]]], [[[0.9542919993400574]]], [[[0.951836884021759]]], [[[1.1712501049041748]]], [[[0.97215336561203]]], [[[1.452519416809082]]], [[[0.9949560165405273]]], [[[1.1478893756866455]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_870a8d14dac98acd801761e8f805c6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3180955648422241]]], [[[1.8995996713638306]]], [[[1.1273280382156372]]], [[[1.849107265472412]]], [[[1.6026138067245483]]], [[[1.8701281547546387]]], [[[1.0979362726211548]]], [[[1.7622225284576416]]], [[[1.2591562271118164]]], [[[1.9011240005493164]]], [[[1.343113899230957]]]], dtype='float32').reshape([11, 1, 1, 1]),
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


    class TestPrimitiveOp_3aceb5209bfb72b97e69935ffe7d4f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22b9e07423eb5b6ae5626ac61207d791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6dd8d15e311df41a73558f7772516a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6ff040b5d0a2c44451901807cc64f05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.4986821413040161]]], [[[1.1676080226898193]]], [[[1.819826602935791]]], [[[1.2086670398712158]]], [[[1.7440937757492065]]], [[[1.4067095518112183]]], [[[1.08768630027771]]], [[[1.2487809658050537]]], [[[1.0397902727127075]]], [[[1.7107311487197876]]], [[[1.6483170986175537]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_2f22e84fcf627fe25b3d1326b0fe4e16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2764a7d4b695a44241591e569f786808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b51e177b2db10f7c50155479125954bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e5918d5d455c524739058e1fd71c232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc9ace5be76dbb1305bc338b5eea46d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e40f7bdb8f807371784f72471a45be6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f65e7df10d88e8e4ba2f81bf2b871dce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43703b6f2df870928ad6759ff68f3c9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b6dfbc4b92eab05043577be50542d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bcc3e551d9172f8c2a16f740b727ece(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.8052839040756226]]], [[[1.666274070739746]]], [[[0.96114581823349]]], [[[1.4701937437057495]]], [[[1.279260277748108]]], [[[1.2332408428192139]]], [[[1.875383734703064]]], [[[1.5806314945220947]]], [[[1.2589390277862549]]], [[[1.6112051010131836]]], [[[1.4541015625]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_44171d977ea3f3e222de37046eb06413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4286ca4ae51c54d83117916378f52a7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cadc06be8f1eb6a1c8f0a887553c58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abaae2980b59f4b62925d21b8b0e18a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12a9454cdec00d64d8cd60eb5861ba08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fc0c4bdd2852aadf6ba1cd321624b91
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()