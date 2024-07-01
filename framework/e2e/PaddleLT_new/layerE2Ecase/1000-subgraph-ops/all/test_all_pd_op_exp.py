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
        paddle.seed(2024)
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
    class PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5531cee36457a5406647430856502ff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.046501144766807556]], [[0.33485233783721924]], [[0.1501895934343338]], [[0.2645980417728424]], [[0.17164172232151031]], [[0.46160778403282166]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_58b7b646a6007b9928963c9d36a5e649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4248183071613312]], [[0.1123322993516922]], [[0.19716881215572357]], [[0.3563118278980255]], [[0.3588925302028656]], [[0.3417365252971649]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_6d500d5c044e021c118ec47c8043eb29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfebfdb7f5a9136d2aabba251fbf9ffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_008b7b809dfdf5e0e1e5c6e6e6942548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf9c23a67404cc965f4a24585ad5d749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2527312934398651]], [[0.2894127666950226]], [[0.4538465738296509]], [[0.2415352314710617]], [[0.06341084837913513]], [[0.28517720103263855]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_5436f64c2833fcaaf7fc93546d77a936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.43791815638542175]], [[0.14317414164543152]], [[0.09789053350687027]], [[0.4941061735153198]], [[0.3782956004142761]], [[0.2173541635274887]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_71ec65f7701db4b083a207e3d40022fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d93119ab16a947245dd6dcea8b73fdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.1501445472240448], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3ed9bdfd539b91b50d92a2cfce25c604(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.36977577209472656], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ee16aad215dc19abc5ced40c0d024fe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea68dd9385ef3b30da666f54f09fac18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02942965365946293], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d55f55b640e1ec8e772282104d3f588c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.040439870208501816], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_36b65fb973d470f47ed1f444243c2b37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4882940351963043], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a160389b8715ed7f6885fd69501b4e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.48973387479782104], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_acae962302fdd51205fa5bac0490b10e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.49126216769218445], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bde9a60e0a19d4bb3b395e40f9722d9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.03385039418935776], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7b67fd40d154b9972b14e0d7c29e7a36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4407913386821747], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3acea216f80dbe334f1f15df286fe477(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3561382293701172], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_97e188db4d9d11b519092934a415a882(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71ec65f7701db4b083a207e3d40022fe
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2267407923936844], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8096c896c467971a75a1e3fa4de88ea2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62051b19fed487ecafded7391e7bdddc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8096c896c467971a75a1e3fa4de88ea2
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9934d76ab83501e7045a5cfe90770805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6426fdadfa727f7fa9640c3471a4e6a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5531cee36457a5406647430856502ff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.046501144766807556]], [[0.33485233783721924]], [[0.1501895934343338]], [[0.2645980417728424]], [[0.17164172232151031]], [[0.46160778403282166]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_58b7b646a6007b9928963c9d36a5e649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4248183071613312]], [[0.1123322993516922]], [[0.19716881215572357]], [[0.3563118278980255]], [[0.3588925302028656]], [[0.3417365252971649]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_6d500d5c044e021c118ec47c8043eb29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfebfdb7f5a9136d2aabba251fbf9ffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_008b7b809dfdf5e0e1e5c6e6e6942548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf9c23a67404cc965f4a24585ad5d749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2527312934398651]], [[0.2894127666950226]], [[0.4538465738296509]], [[0.2415352314710617]], [[0.06341084837913513]], [[0.28517720103263855]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_5436f64c2833fcaaf7fc93546d77a936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.43791815638542175]], [[0.14317414164543152]], [[0.09789053350687027]], [[0.4941061735153198]], [[0.3782956004142761]], [[0.2173541635274887]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a17a1cc4bf5449828ecd8fced2409770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.1501445472240448], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_90b4da04335e61aca9e37d4d7d39dc09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.36977577209472656], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ee16aad215dc19abc5ced40c0d024fe2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca2d6b8bcb028176a82f9115061d2fdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02942965365946293], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cd0236c30def0fa8ddc33a022dc917da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.040439870208501816], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e2bda7ce4eda8eb57f48ee4d905c30dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4882940351963043], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0aa81dc870d3b7402c50377d4cce169a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.48973387479782104], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8fe1455afdf0309556dab6807af92744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.49126216769218445], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4714de3c56b7ebdd0db31a6d2808e63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.03385039418935776], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8eae24ffcc086fa005d51ef418e546c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4407913386821747], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8916d105e8979284d3307cb8c514f9b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3561382293701172], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4bc35d6ea09cbde0528f292c9a37ed0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_171e2c45b9d82a5b5c95261693ed1370
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2267407923936844], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62051b19fed487ecafded7391e7bdddc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8096c896c467971a75a1e3fa4de88ea2
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9934d76ab83501e7045a5cfe90770805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6426fdadfa727f7fa9640c3471a4e6a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daa040e225658ca1eaf6be4b9e1ef125
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()