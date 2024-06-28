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
    class PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52cb3d768f03a43073b762c9e1cfd6af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_48ae8293522225db444a3779ae6a8519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d963be664156ba4e5e845042068a465b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_248956547b219d3e9f5aa542f68dc877(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_75b0cb90e4370965b641cfa2ddb606e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0e9538fff6b3416216480c5a8bf12f0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_32dc87ac0b61d5139849882a3f265af4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_507ce55d5fbfbde7b70f24f041ec11c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_84f89adad51485146cd8c18c139af962(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bd4ad446fe6bc0111afdc0f7e663417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9fb58fab53054f3ea48bb64fb05a00a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1091941369d52e0487b5adb664bd103a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1091941369d52e0487b5adb664bd103a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5401fb772bb358cb9284f625c2368b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1bd4ad446fe6bc0111afdc0f7e663417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f25af2447c082bad90c1d41cb79e930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2807962894439697], [2.2598795890808105], [1.870574712753296], [2.1764073371887207], [1.995255470275879], [2.09159779548645], [2.2656216621398926], [2.2172205448150635], [1.9983623027801514], [2.1377906799316406], [2.196044445037842], [2.0789554119110107], [2.1784939765930176], [2.2193682193756104], [2.052372694015503], [2.2238101959228516]], dtype='float32').reshape([16, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3e7fd739ede6fbeeed3841887caa94d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.960141658782959], [2.1533260345458984], [2.1658780574798584], [2.2659358978271484], [2.0110549926757812], [2.075639009475708], [2.0796830654144287], [1.8293005228042603], [2.01572847366333], [2.2285847663879395], [2.2727530002593994], [1.9443649053573608], [2.290123701095581], [1.808598279953003], [1.9616131782531738], [1.8629130125045776]], dtype='float32').reshape([16, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a18d056eb4e5640bb2383cb52e2c707a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_45a64af6215885a09230b0fc03d29874(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bd4e92e1a1689f351c6dc60c3e29675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1bd4e92e1a1689f351c6dc60c3e29675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_74ee1b87bbeb54dacad4ff57cf7891e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d362e3c1666c0a881c6d241b57b3912a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ee1b87bbeb54dacad4ff57cf7891e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a34d9fbf41e1b17aad4794985e4f0d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ee8800a18eb8dc0231fce37d041d036a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_55583be25ae0e5b84d01c4fadecddc9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_20c713834aa62741d180afe34aa60b5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32f71412550c2450525cc46fc144e57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_32f71412550c2450525cc46fc144e57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6d9254c786c018ffd8c8a17054e0fff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_39c828e4ef4838ede21fdfd92d4e3488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_de0c2214953a82af913b8d8b55116cda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac1329af31959853322c63526743b6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de0c2214953a82af913b8d8b55116cda
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_507ce55d5fbfbde7b70f24f041ec11c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cedd59113a265722273207af58e27253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b90753aac6b965dbee79cbcd6104a4d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b90753aac6b965dbee79cbcd6104a4d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9058f0225995b64b47c1ceeba6361bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9058f0225995b64b47c1ceeba6361bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c2cfb8bcb424e81236fd514e5774ef8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba2b9da5d1aa1b47442710e8f6275ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cfb8bcb424e81236fd514e5774ef8b
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0e9538fff6b3416216480c5a8bf12f0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e7bcb123e1d012f9a46bef6fb65a9c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e7bcb123e1d012f9a46bef6fb65a9c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6a553487c260d26a18ce6449489f824e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e2c3011437f7caa89d6adfde302483ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d019a3d38159034bec945f75d3686ceb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cfb8bcb424e81236fd514e5774ef8b
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_606ce8daa1b982c5096de4f44340df58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_958e45a0464d0b72576a1dce12f5a99e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_958e45a0464d0b72576a1dce12f5a99e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7b5d8216032038f0133658b93ab35cd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a593859bacecca89374701a516c41574(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_59ccd735b11871f70e6b2561a499c451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_39c828e4ef4838ede21fdfd92d4e3488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_94344f4dc0874442b76bc7ce1171c32a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8b71d60fa1fe9753c5fa6b90c1e8817f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2159955501556396], [2.2937707901000977], [1.9798015356063843], [1.930498719215393], [2.3208510875701904], [2.086238145828247], [2.125875949859619], [2.1606414318084717], [2.064732551574707], [2.2746167182922363], [1.9600178003311157], [1.9090816974639893], [2.188173294067383], [2.0657029151916504], [2.205883741378784], [2.1968834400177], [2.008652687072754], [1.8923864364624023], [2.230451822280884], [2.1479268074035645], [1.949690341949463], [1.9484350681304932], [2.20443058013916], [2.192775249481201]], dtype='float32').reshape([24, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3c784d54cb3833226ae6c8fa9b5d38bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.1657958030700684], [2.0923051834106445], [2.312364339828491], [2.0151829719543457], [2.0399513244628906], [2.1100361347198486], [2.1869425773620605], [2.2116706371307373], [1.970133900642395], [1.8513662815093994], [2.2078945636749268], [2.220553159713745], [2.2786765098571777], [1.9406858682632446], [2.060137987136841], [1.9701740741729736], [2.385483741760254], [2.1726832389831543], [1.9692106246948242], [2.2667107582092285], [2.135495185852051], [2.080367088317871], [1.9795235395431519], [1.9408769607543945]], dtype='float32').reshape([24, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2db805c58f41c876234fc00188829cb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e6323a4a17e64dae18929284de7a603d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_99459edd58e27fe33edb0f129835002f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_99459edd58e27fe33edb0f129835002f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d006e0545b2d5a519b8109478b0e80b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_16414881397166d3516d31c90db074a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_10d7179e2015a8b1b3a75140b9a1ee6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.888964295387268], [2.2693610191345215], [1.9160085916519165], [2.2557058334350586]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_59525440014a70bf8293b2b1bd99cd81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.8958507776260376], [2.0913102626800537], [2.2948904037475586], [2.1589481830596924]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_370bc1ddeedfade45d85fbef51a19df6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ee1b87bbeb54dacad4ff57cf7891e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8de8597909937eee3e81cd4b122c47c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e23fe0de54de14459692bc366985642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8de8597909937eee3e81cd4b122c47c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a8723ff1a1b96baa59ea1f37f6bbb485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9bbb51fd14be9007c3d0faf506b97f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c9a434605cf12c3a730cdadd83d92965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ee1b87bbeb54dacad4ff57cf7891e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3efb849d79c44d503164642f577663f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cfb8bcb424e81236fd514e5774ef8b
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2ec9b80b4db82a50ab2a59a3ad93447b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ebcd4653f0c54091de811e5e48d2471b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ebcd4653f0c54091de811e5e48d2471b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a8e2f8ba4b941f6f1b54807d6bb7b867(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0d39a430f85810884178555548194f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9ddb0f1b61860fc72647b46f23242bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7b5d8216032038f0133658b93ab35cd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1fb15f3976a08ada9ce46f7c22119cdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d98c6e74139e0c730a7141fd25c55679(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7af3fbf27b8e671a9e2b7ffb56c21d8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f157f378f738d6968dbe4fd9ac15109b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f157f378f738d6968dbe4fd9ac15109b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_384606a079d79d0cb703725338375294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5b962d01c624e8d153f44be4b17c7b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5b962d01c624e8d153f44be4b17c7b6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_97c453dcdf1e420192f029a0a465fbeb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36f8c60655fd7f21e91bcfcaf22a64f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97c453dcdf1e420192f029a0a465fbeb
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_218665c6a1e4226d5a8e86ebafa9e1de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07e55b07b1e5523a9cd006f5946bbe21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_218665c6a1e4226d5a8e86ebafa9e1de
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f7afe5657c6fd5b52a91885d3d174b1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9977fb9861085d022bad4fce09141cde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7afe5657c6fd5b52a91885d3d174b1b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1291a886960e88c743481d2a8a74c4e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1291a886960e88c743481d2a8a74c4e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a21a19bea1a0c427c8cc4cc86f570293(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bda8538b5d57294f9c68b101a6f921f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de0c2214953a82af913b8d8b55116cda
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_69a696d961b2c28368a19a8afc09e0c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ee1b87bbeb54dacad4ff57cf7891e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3ad3014ab60ac25fba1daadc23a74e7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7c55cd3abdd34aac44fd47e4339c172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad3014ab60ac25fba1daadc23a74e7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_78442d3517aefdf3ac06080380c3f2ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b888e3cf96ad8b86c3fd2396b362a011(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78442d3517aefdf3ac06080380c3f2ae
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b9b318a79efeb518ad66ebe4e3be817a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3c3cd627468a02b324a7f6a2e3b73cf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_18f357b291d67c2ed3ae90823cbd0074(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d26e0c1627996804db07b86452e4e74d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1df8309943ad76b9b2f07cfe933ac742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1df8309943ad76b9b2f07cfe933ac742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9f19da50e3ab9594726db95a4c0ece31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9f19da50e3ab9594726db95a4c0ece31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a990cd64dfe6577aeb943fc01bb03d76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a990cd64dfe6577aeb943fc01bb03d76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b30e5a3d39dff8800f4741d044620577(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b30e5a3d39dff8800f4741d044620577(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5359bfea736e14804c65d77a81a4a569(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5359bfea736e14804c65d77a81a4a569(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e4c913b5d866888740a0002c0192ee6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e4c913b5d866888740a0002c0192ee6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fc003a8f21cc838926e378b4db5ada59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_032468f9792ff8aa6f23131cf43aa242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_032468f9792ff8aa6f23131cf43aa242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a70bfafbcf3ca5ae521e4cfc7e9c12a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_58964f7c5e9593ac8b1e228fa6392667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_55583be25ae0e5b84d01c4fadecddc9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_8b6725f254b5ab0d0b92e7c3580e42b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5bcb88564b676ebe44889accc3319dc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_f95e38fe23cdb9e66272940e8555a739(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97c453dcdf1e420192f029a0a465fbeb
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c0ad80a74dcc2c87ad56233804520971(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_6d971aaf44c0bdbf2a4cdc7f58dd3808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_2efdc92b575dfdb8a6319590d782a713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.155951499938965], [1.864601731300354], [2.0482661724090576], [2.106358766555786], [2.1773722171783447], [2.072309970855713], [2.1956989765167236], [2.012533664703369], [2.229339122772217], [1.858726143836975], [2.2677783966064453], [2.1236958503723145], [2.0704946517944336], [2.188840389251709], [2.073134422302246], [2.157585620880127], [2.0383729934692383], [2.3608081340789795], [2.0037755966186523], [2.065575361251831]], dtype='float32').reshape([20, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a5b17567927aadd27aade65abe68714e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd09ab5354cf028ae3c58ec448e990b3
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.052689790725708], [2.2076961994171143], [2.0857365131378174], [1.846848964691162], [1.8959813117980957], [2.0812926292419434], [2.1710519790649414], [2.272707462310791], [2.07831072807312], [2.199514627456665], [2.1488406658172607], [2.2427282333374023], [2.193753480911255], [1.973362922668457], [2.1430397033691406], [2.0266079902648926], [2.1450860500335693], [2.1068367958068848], [2.3094241619110107], [1.8611677885055542]], dtype='float32').reshape([20, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1fb15f3976a08ada9ce46f7c22119cdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0d39a430f85810884178555548194f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1bd4ad446fe6bc0111afdc0f7e663417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_03702ea7b3ebcd420a996246d013eb8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_218665c6a1e4226d5a8e86ebafa9e1de
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f4f61cddd1f3d9ba2875899418f882d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7afe5657c6fd5b52a91885d3d174b1b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7af3fbf27b8e671a9e2b7ffb56c21d8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_028d85a84f3e6744e75232499140ee54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_028d85a84f3e6744e75232499140ee54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d9c19991f7800db37abf1877629d74a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2cfb8bcb424e81236fd514e5774ef8b
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a8e2f8ba4b941f6f1b54807d6bb7b867(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9bdb4d7fa6ab1868be068f3b7bcb835
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d63bd01deb29f8459c94115fcf62f9f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32dc87ac0b61d5139849882a3f265af4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a098e1a3f9346022af4c944c3e13ef81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_11f050d26dbdf2f1c0897690f7528af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_11f050d26dbdf2f1c0897690f7528af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20c713834aa62741d180afe34aa60b5e
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4f4f6b3a1952cde8265404035b4bc935(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_327f1a34a59c0a9d22e4de6469cd831e
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7ab2433d3397d915444f000233d4229b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad3014ab60ac25fba1daadc23a74e7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1479070bac25b763303f13618ecf0f56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78442d3517aefdf3ac06080380c3f2ae
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_1326c755e33d493195c449573760290d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc7f219cb190cda5a7d2a8d2447893b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1326c755e33d493195c449573760290d
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_cfda6b0072a52fa391798560f7943427(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 92, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc47fa3d99ee74ccfd3d0e3fae4be57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfda6b0072a52fa391798560f7943427
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5319f0bc7de0be7a5b5c5f4bbce80a36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2048, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4576ed51f395f819a38cb0714e97550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5319f0bc7de0be7a5b5c5f4bbce80a36
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6faa23f4eabb0d5c90ba62a6e470ec40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ef95e625257881694031afa77bb363c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6faa23f4eabb0d5c90ba62a6e470ec40
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0ff5f2df11a95eb1bc8b6bb3088200ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d3433807a1e796017b55eadff3bea46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ff5f2df11a95eb1bc8b6bb3088200ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e355bf5b444b339bf0a839b812a4573a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 336, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0bd6b7c09cdfff8b6f04b0698175f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e355bf5b444b339bf0a839b812a4573a
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_5949ba36e735d5b9a6af51baf4bfe60a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 3549, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89bbc945f8cb6ab7b5c567dc64f0b1fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5949ba36e735d5b9a6af51baf4bfe60a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3d7ac4c511d8e98c23897e782c5a281c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 60, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf88e81e28359c2dbcec2286eeac6b73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d7ac4c511d8e98c23897e782c5a281c
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_eb163c3f68e712a11e3aa8ce59b63832(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_882fe94a31c9d26a4cbc52aa23aac73d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb163c3f68e712a11e3aa8ce59b63832
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9973fc4226bf7b2330ec489f5e1885eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39b642ccbcd8b71b54bd752eb3cf6e3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9973fc4226bf7b2330ec489f5e1885eb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4569043c775ebccd8627e7de58841833(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 336, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30e8a6b9940fb9a117cf66030cb6d6d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4569043c775ebccd8627e7de58841833
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_30e8a6b9940fb9a117cf66030cb6d6d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4569043c775ebccd8627e7de58841833
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7e2113855e419c5d090802c7196c02d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2955ec0a6d2f0db4b43b26ec6da9fd88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e2113855e419c5d090802c7196c02d6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_882fe94a31c9d26a4cbc52aa23aac73d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb163c3f68e712a11e3aa8ce59b63832
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7a3f8daf4018628d776ac1424a817f51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d63f82143da726a38fbad0d502ca9f3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a3f8daf4018628d776ac1424a817f51
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2807962894439697], [2.2598795890808105], [1.870574712753296], [2.1764073371887207], [1.995255470275879], [2.09159779548645], [2.2656216621398926], [2.2172205448150635], [1.9983623027801514], [2.1377906799316406], [2.196044445037842], [2.0789554119110107], [2.1784939765930176], [2.2193682193756104], [2.052372694015503], [2.2238101959228516]], dtype='float32').reshape([16, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_01712ad39db7c892850f928727eef50a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a3f8daf4018628d776ac1424a817f51
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.960141658782959], [2.1533260345458984], [2.1658780574798584], [2.2659358978271484], [2.0110549926757812], [2.075639009475708], [2.0796830654144287], [1.8293005228042603], [2.01572847366333], [2.2285847663879395], [2.2727530002593994], [1.9443649053573608], [2.290123701095581], [1.808598279953003], [1.9616131782531738], [1.8629130125045776]], dtype='float32').reshape([16, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ef8e2a32122827eef8c0b04313e8b7f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 240, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_adf944244a82527f1eabf9085dabf1d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef8e2a32122827eef8c0b04313e8b7f0
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7854dfd96c7be0cb5c0af37fda0dbd67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 7581, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19641d8fbe282ad43fc03eece4136f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7854dfd96c7be0cb5c0af37fda0dbd67
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_61588e983fc6eb03c5c264e3e2853a0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 18, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b44c1f35e9f44c7d1a89b535ac178bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61588e983fc6eb03c5c264e3e2853a0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7b44c1f35e9f44c7d1a89b535ac178bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61588e983fc6eb03c5c264e3e2853a0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_918f6a55d915194211ec9dab4c2820b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 66, 130], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_484f3ea251492a3315c29f5c36a21383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_918f6a55d915194211ec9dab4c2820b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_98850ac5761b987cc28e90bb6715bb8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 4725, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c75297d91f124ce0a178d58476f08e63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98850ac5761b987cc28e90bb6715bb8c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a6cf8c3eb381f2e473305192c52d425b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 60, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c0a4567da71503267f7e21df201bf8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6cf8c3eb381f2e473305192c52d425b
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ee1246826b24bc9d681e78f6d37f886d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 872, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5708bb082619be412daf0b27a6112aa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee1246826b24bc9d681e78f6d37f886d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_1a4c6b6bceaa7a2aea8d872342d43ede(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d4e5eee6ea0e70dc0adfcc62143d93d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a4c6b6bceaa7a2aea8d872342d43ede
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5d4e5eee6ea0e70dc0adfcc62143d93d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a4c6b6bceaa7a2aea8d872342d43ede
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f03aa9f1d4340176f6043f2cc5878a2c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 8400, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_483f4ddb3d9e70232ad825f213460557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f03aa9f1d4340176f6043f2cc5878a2c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_71a1cbb7c57b0a6060ed07a87ca8362d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 336, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d8c487a576f5777995120cd30b061976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71a1cbb7c57b0a6060ed07a87ca8362d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_75e04e41f9f02b60977453a531c69fa6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8470c3327185952cc2f0d2b3333472a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75e04e41f9f02b60977453a531c69fa6
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_89bbc945f8cb6ab7b5c567dc64f0b1fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5949ba36e735d5b9a6af51baf4bfe60a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3a17bdc924bde25fa8899b47b4dd3cc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 240, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55eff967e67815eae6ce129470a954a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a17bdc924bde25fa8899b47b4dd3cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_032e105b46352a00dfda113986bf826a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e22a3de99550fe0fc2e2c2ecf2895ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_032e105b46352a00dfda113986bf826a
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9e22a3de99550fe0fc2e2c2ecf2895ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_032e105b46352a00dfda113986bf826a
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_2482e3d90994cdc7d467f74deed5f1d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d4d69fedd76bef175bf1a1650c49950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2482e3d90994cdc7d467f74deed5f1d6
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7d4d69fedd76bef175bf1a1650c49950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2482e3d90994cdc7d467f74deed5f1d6
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_712ee52e5b0914988b3200bd2a9830c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1000, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_406b9a1c8ff59880bcd2d002a5b7f386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_712ee52e5b0914988b3200bd2a9830c0
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e0bd6b7c09cdfff8b6f04b0698175f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e355bf5b444b339bf0a839b812a4573a
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_86bf2319dd48ebf82fce9ecaa68c6f96(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b478018fcc36df2b34c9b89fa2924ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86bf2319dd48ebf82fce9ecaa68c6f96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b478018fcc36df2b34c9b89fa2924ce6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86bf2319dd48ebf82fce9ecaa68c6f96
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_bcbedb470b95a5dd773a484b8898efe4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 36, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20bc9349080b012c272e50f2f9733742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcbedb470b95a5dd773a484b8898efe4
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_27a961658cb36e846da180ce097f20ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1280, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c58492e1d1824d5e2cf6a5313bca1518(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27a961658cb36e846da180ce097f20ff
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8864e6cdc6cadc0580de9c61705cd1fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_846caed4cb7949557ecd4965b85872e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8864e6cdc6cadc0580de9c61705cd1fc
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_ea41449c4ce8950f34b6b3e046f9322c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 480, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8801fb0bc10b5c9e1a22dffc0f88fa6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea41449c4ce8950f34b6b3e046f9322c
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_590cffde91e2a6c1a5fc999940ca4aea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3651f30b65a1fd3bb5aa40d4c7f8445(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_590cffde91e2a6c1a5fc999940ca4aea
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b3651f30b65a1fd3bb5aa40d4c7f8445(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_590cffde91e2a6c1a5fc999940ca4aea
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_634af8846d02ad32c6d3c715bce3f758(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 336, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8661d29c8e60e208598f94187ea32b29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_634af8846d02ad32c6d3c715bce3f758
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e62255ea3dd0098ba4c4623ca2886264(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 4116, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a3a6525fe473ca2ebaaad45baa90197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e62255ea3dd0098ba4c4623ca2886264
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ef65e18202ac12d8cdc134e453b281d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 240, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44f1a2b9de7c61d6fe81199d3fec9e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef65e18202ac12d8cdc134e453b281d5
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_d8c487a576f5777995120cd30b061976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71a1cbb7c57b0a6060ed07a87ca8362d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7bdb0a19702881c4bca96954db1c8028(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1536, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62b7e2c6873b655da0f45b7b2d37d474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bdb0a19702881c4bca96954db1c8028
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_959c6339d2e2a622328a8f6a7a4dc5d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91cab69442899bf0c9c03f799d114390(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_959c6339d2e2a622328a8f6a7a4dc5d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2159955501556396], [2.2937707901000977], [1.9798015356063843], [1.930498719215393], [2.3208510875701904], [2.086238145828247], [2.125875949859619], [2.1606414318084717], [2.064732551574707], [2.2746167182922363], [1.9600178003311157], [1.9090816974639893], [2.188173294067383], [2.0657029151916504], [2.205883741378784], [2.1968834400177], [2.008652687072754], [1.8923864364624023], [2.230451822280884], [2.1479268074035645], [1.949690341949463], [1.9484350681304932], [2.20443058013916], [2.192775249481201]], dtype='float32').reshape([24, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fa29a9c7412fb341fcb408dced041d7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_959c6339d2e2a622328a8f6a7a4dc5d6
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.1657958030700684], [2.0923051834106445], [2.312364339828491], [2.0151829719543457], [2.0399513244628906], [2.1100361347198486], [2.1869425773620605], [2.2116706371307373], [1.970133900642395], [1.8513662815093994], [2.2078945636749268], [2.220553159713745], [2.2786765098571777], [1.9406858682632446], [2.060137987136841], [1.9701740741729736], [2.385483741760254], [2.1726832389831543], [1.9692106246948242], [2.2667107582092285], [2.135495185852051], [2.080367088317871], [1.9795235395431519], [1.9408769607543945]], dtype='float32').reshape([24, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_b72319f5756483734e4ad9109ec77fea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 60, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2bd16d67f9eee52d44f5850c9199466b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b72319f5756483734e4ad9109ec77fea
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a7c71d32f1024462cbf7081d8817bb0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 6069, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_550b0fdb58da453714ebc45d101d59c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c71d32f1024462cbf7081d8817bb0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d81502faafc18b7417f3dc48548296a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6deab7ed028de24eb642318563572483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d81502faafc18b7417f3dc48548296a5
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6deab7ed028de24eb642318563572483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d81502faafc18b7417f3dc48548296a5
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_821e02df4827c1eed7bb5b44d7f5c8f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 240, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_927a7d8b4897a574c5bc17c4f20a2c8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_821e02df4827c1eed7bb5b44d7f5c8f2
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_017d0b939549b5040e5bdd72b804dbaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1536, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fa545d2f3e6381b8908e9d3f78535dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_017d0b939549b5040e5bdd72b804dbaa
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8b6439b9ffccb41e8836f902ae42766e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c132829db52e7029a0e9459d64d71224(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b6439b9ffccb41e8836f902ae42766e
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.888964295387268], [2.2693610191345215], [1.9160085916519165], [2.2557058334350586]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ad61c3833acd7e94c3294191f1576b97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b6439b9ffccb41e8836f902ae42766e
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.8958507776260376], [2.0913102626800537], [2.2948904037475586], [2.1589481830596924]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_fe38f2ac5061c6d9b987bbb6c7ba6e1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 70, 134], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7adbd4fd0205271a0e959a081ee8963d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe38f2ac5061c6d9b987bbb6c7ba6e1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_5cedc036ebc3778c96c61ee237c7aa68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 104, 101], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f071297cabc74e99930966d025250d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cedc036ebc3778c96c61ee237c7aa68
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_819bc6d962b49c2f1733ba9fd321033c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_757d02da8260e2a692c82f7df3b41a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_819bc6d962b49c2f1733ba9fd321033c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_bd973336be6806b8201c0be9005d8436(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 36, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55cc568cef529dfd72fb0f64ea3a2089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd973336be6806b8201c0be9005d8436
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_191159b028989c195a389d5afdc446eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 68, 132], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6081fa1fbf0733063a096c3811db8cbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_191159b028989c195a389d5afdc446eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c6cb7165769b155d07822222953a7fa1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1000, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42c408cf4b6380195ce09c45baf6df8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6cb7165769b155d07822222953a7fa1
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_b7c4005ee3b1ba4d87ff3000b7ef4da0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 60, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aed118969644688cd70dc2bff279fe61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7c4005ee3b1ba4d87ff3000b7ef4da0
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6eaa84f05936de926834364c37d951f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 36, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54e679ef01e02566440fb67f8c6afb57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6eaa84f05936de926834364c37d951f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_54e679ef01e02566440fb67f8c6afb57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6eaa84f05936de926834364c37d951f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9b8798dc159e660a2eaa0ecb4ec7518e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9299f76dfa4fcee9d133c204d16cd6dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b8798dc159e660a2eaa0ecb4ec7518e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7052195a601d6966e56faf22ceab333d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2014fc4297ab3b8a36549fc197ea09d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7052195a601d6966e56faf22ceab333d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7a00e9407cd2fa5ba67de7f1fe647b08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6dfa9645799616f5e021585c5cb2ea6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a00e9407cd2fa5ba67de7f1fe647b08
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8661d29c8e60e208598f94187ea32b29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_634af8846d02ad32c6d3c715bce3f758
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_da78be35576b20d51fe92e7c693be5b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a0e6261cb31c2aaf2905e7c15797db7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da78be35576b20d51fe92e7c693be5b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8794fea576028270ace6101e855b91b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2048, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a650baf3fcc8597f41aa18997666af76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8794fea576028270ace6101e855b91b9
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_a35b0c8e7f1a739adb37cf033ef5ce38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_393ee9dad454ce68a7cf3e34f3f2b4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a35b0c8e7f1a739adb37cf033ef5ce38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_6d97d4da71b98e74ce36bc36290a3624(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b05478a0b5d9dc56654e90d2cd77994a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d97d4da71b98e74ce36bc36290a3624
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b05478a0b5d9dc56654e90d2cd77994a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d97d4da71b98e74ce36bc36290a3624
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_40aba78d845296b9c3d9cbfd61c41da0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74c6228b306b6b2deb8d937dfdd1a878(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40aba78d845296b9c3d9cbfd61c41da0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8e1fa789c7fc33e31808ef759d8ab064(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8909de80378bb96f3640edaea8bff9d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e1fa789c7fc33e31808ef759d8ab064
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8909de80378bb96f3640edaea8bff9d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e1fa789c7fc33e31808ef759d8ab064
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_95738f1fc43bcaa7fff942fde09e1dca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 96, 1, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6d7e171035d7a58b96193b9f267c937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95738f1fc43bcaa7fff942fde09e1dca
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9a7c2691642945ccae87e560fac85a06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fd074b5d75515709df1606bac0e35bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a7c2691642945ccae87e560fac85a06
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_626d37b9a356dab894510182a1cbf9fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_935952c211c7360d7e492be97023b054(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_626d37b9a356dab894510182a1cbf9fa
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c0d42d32b08eae90968c32979683b71d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ffeaa321ebe224367b8a3dd1680649de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0d42d32b08eae90968c32979683b71d
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ffeaa321ebe224367b8a3dd1680649de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0d42d32b08eae90968c32979683b71d
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a168d6f843b0e7ca9e55548d2a3d9970(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 9261, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8b3e35b9a3d3e944c3bf99412808e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a168d6f843b0e7ca9e55548d2a3d9970
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_75f35e1819202a3d4b00cff0008dbdf9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6075a7ec31f803e1cd61cb82a1e6e9f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75f35e1819202a3d4b00cff0008dbdf9
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_bfaa7e724b6cf573592d69a7233e6e35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdb640ffd26cac7f0bdbe500bb3da6c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfaa7e724b6cf573592d69a7233e6e35
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_62b4f5773327862dd11a2e27d75a7a9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_384e64a28a9a7683054e2cd8936c9b4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62b4f5773327862dd11a2e27d75a7a9f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3d8b1359a9a927512de62f98ef3a85a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1000, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_642c945c964fd20aaf35b63b06f595ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d8b1359a9a927512de62f98ef3a85a9
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_cdc76da96c468b6e7996cc9d08098593(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 2100, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45000b145177a3c2e0083b90c5555949(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdc76da96c468b6e7996cc9d08098593
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4707f56ef62413be289b889d9416786a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1248, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91c523cd6df1d65d24c990f8edc2866c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4707f56ef62413be289b889d9416786a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_7616a4d54e468d4356bae41713f3bbbc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 480, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d04896d58ecbd5ed368c84666f8a4076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7616a4d54e468d4356bae41713f3bbbc
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_13a160c452a9f4749e486ccf87d6c2fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 36, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_500719b8929d22c63e6dbc3cf4a328df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13a160c452a9f4749e486ccf87d6c2fe
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_436b361565d0dd1ba2cd3955c61c624c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 9, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccfb560b91d90610feb5a4dbcc6fd519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_436b361565d0dd1ba2cd3955c61c624c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ccfb560b91d90610feb5a4dbcc6fd519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_436b361565d0dd1ba2cd3955c61c624c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d4d54c16190e98ac6441bb6e87933fb4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d38a91095041668bf9a46dbd393b8742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4d54c16190e98ac6441bb6e87933fb4
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d38a91095041668bf9a46dbd393b8742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4d54c16190e98ac6441bb6e87933fb4
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_b2069a6fc9cf8e8126e351d82db3236b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 96, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcc98dc6d7495e97914f6d0e707c5bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2069a6fc9cf8e8126e351d82db3236b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fcc98dc6d7495e97914f6d0e707c5bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2069a6fc9cf8e8126e351d82db3236b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_60a70058e20b840752a6771221a83934(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f837192aed6896cbce8108e968d56005(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60a70058e20b840752a6771221a83934
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f837192aed6896cbce8108e968d56005(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60a70058e20b840752a6771221a83934
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c257256dfffa0fab1b0a707d7a62a322(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c475911098e24299dc122d728b23b917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c257256dfffa0fab1b0a707d7a62a322
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c475911098e24299dc122d728b23b917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c257256dfffa0fab1b0a707d7a62a322
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_5ff3f88b40ad86751aa0dd5b0509dd4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 24, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2cd02262108eb0ae28377d6816d63b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ff3f88b40ad86751aa0dd5b0509dd4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c2cd02262108eb0ae28377d6816d63b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ff3f88b40ad86751aa0dd5b0509dd4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_d12c2e8fe92ca1b724c729eaf4249fcc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 156, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed8be118b04c40b59e88ad7c706475b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d12c2e8fe92ca1b724c729eaf4249fcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_c4d1f592c8fc76a4ccd323fae4ee3fbf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 48, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9468efef319d0accb81e880862ec6813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d1f592c8fc76a4ccd323fae4ee3fbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9468efef319d0accb81e880862ec6813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4d1f592c8fc76a4ccd323fae4ee3fbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_a988152bb2046d18f75683a0fee817b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 11109, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_323062f1cc2b9cea7ed5a951000a2452(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a988152bb2046d18f75683a0fee817b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5708bb082619be412daf0b27a6112aa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee1246826b24bc9d681e78f6d37f886d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_28aa92f1b91d8a1479eb7dbf907765f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 480, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31a9fc7e4dde50c3cf98144a5329666f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28aa92f1b91d8a1479eb7dbf907765f2
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_0d52262f300aaa484915f7a33816d3e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 480, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea7b761f5cfc7ec58bf23aa4db9815fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d52262f300aaa484915f7a33816d3e2
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6c7bc0dd9bcf5904ce469bc5d0f29828(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 192, 1, 25], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_850a91d59d4cf5c238374ee704102d6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c7bc0dd9bcf5904ce469bc5d0f29828
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f3f6f02564338326646b6bf6dad86d9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 36, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20766dd9f1c25019f645a1633bff9335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3f6f02564338326646b6bf6dad86d9a
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_6dfc92b0395d27ebac08c100d1b1a142(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cee21842ca6fe78d3c134f8a71fe619(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6dfc92b0395d27ebac08c100d1b1a142
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_8de1c6ab7072211dbf8642eab9a87bd0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d60f4cdb27eac95ff1f2b35b88ee762a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8de1c6ab7072211dbf8642eab9a87bd0
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.155951499938965], [1.864601731300354], [2.0482661724090576], [2.106358766555786], [2.1773722171783447], [2.072309970855713], [2.1956989765167236], [2.012533664703369], [2.229339122772217], [1.858726143836975], [2.2677783966064453], [2.1236958503723145], [2.0704946517944336], [2.188840389251709], [2.073134422302246], [2.157585620880127], [2.0383729934692383], [2.3608081340789795], [2.0037755966186523], [2.065575361251831]], dtype='float32').reshape([20, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_349c74a371988db22ff4749e031e76ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8de1c6ab7072211dbf8642eab9a87bd0
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.052689790725708], [2.2076961994171143], [2.0857365131378174], [1.846848964691162], [1.8959813117980957], [2.0812926292419434], [2.1710519790649414], [2.272707462310791], [2.07831072807312], [2.199514627456665], [2.1488406658172607], [2.2427282333374023], [2.193753480911255], [1.973362922668457], [2.1430397033691406], [2.0266079902648926], [2.1450860500335693], [2.1068367958068848], [2.3094241619110107], [1.8611677885055542]], dtype='float32').reshape([20, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2a0e6261cb31c2aaf2905e7c15797db7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da78be35576b20d51fe92e7c693be5b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a2014fc4297ab3b8a36549fc197ea09d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7052195a601d6966e56faf22ceab333d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_882fe94a31c9d26a4cbc52aa23aac73d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb163c3f68e712a11e3aa8ce59b63832
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c6d9e0e17116c7fab07e01a3f1e6f1ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69f87a889f5bb80d59f835ce674fdb0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d9e0e17116c7fab07e01a3f1e6f1ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_bc37dddec4073cbff89254cc0f2ee508(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='int64'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8074af2b8fea1afba14583041a22f2e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc37dddec4073cbff89254cc0f2ee508
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_393ee9dad454ce68a7cf3e34f3f2b4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a35b0c8e7f1a739adb37cf033ef5ce38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c319bee66474c4cbd8d0d56f7626e898(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_828d2b4ccd8f80fa0d2f1ab8bd8ad771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c319bee66474c4cbd8d0d56f7626e898
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_828d2b4ccd8f80fa0d2f1ab8bd8ad771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c319bee66474c4cbd8d0d56f7626e898
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_1a671006d1debe1fe8d8efcbbbaf28d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1000, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d50c95b785ba029859b1f9077c35b331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a671006d1debe1fe8d8efcbbbaf28d0
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9299f76dfa4fcee9d133c204d16cd6dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b8798dc159e660a2eaa0ecb4ec7518e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_3c55df83cd7f9cec0d487fb0789abebb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 3024, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9522e5c9b945240132824d4c0513c411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c55df83cd7f9cec0d487fb0789abebb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8ed8abedd4c48674aa8abeef4b2c1888(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1280, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5cec7da6e8bde5af3b8a1472d0e7349e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ed8abedd4c48674aa8abeef4b2c1888
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_bafaa1881b979148514e8a47462c9cd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b022cd255da95b25c9159ba16dc4e899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bafaa1881b979148514e8a47462c9cd8
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b022cd255da95b25c9159ba16dc4e899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bafaa1881b979148514e8a47462c9cd8
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_36b97346ef9635e97f8606b7b5beb525(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 624, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78d507a49fd5962b4b7e72207f25b8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b97346ef9635e97f8606b7b5beb525
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_71a198dfbc293a917ab56e441d2e9663(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1000, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3aa2ec497daf31523c0aad29246219f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71a198dfbc293a917ab56e441d2e9663
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ef791f0d0a3373a886c51538e105c7b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1000, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_650e876ba1c11d6fc6c4bb5b2107d6bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef791f0d0a3373a886c51538e105c7b0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_127150ed89d10ae71181882049044e03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_393b4ebad73f0f275e23ed2ac12ce63b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_15dc843b02f2e8dd9c40c38fcd89bcf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e60d72b731820d29571603fc189b0952(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_fb93099981c19b97d64e3cf273794e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4393c3a4ff1a4d89e08f1333a2290624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e9909792fda9f1c7a9eb6227caa277e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e45072e21be0f0b041e382ab78bf2722(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25ef81f1b8ab38a24842dcd704672d9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_74cd3124a12a801a9e6a456df24eafa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_500609b1203ae1baac72b17eb7727a14(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21f3713dccccb97b4eb1069dcfc4b772(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_931b6f60ced89043ecb1aa4ad95d2e83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_75eb003b7071c336f85a2b6da4d84c3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_75eb003b7071c336f85a2b6da4d84c3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ca485202c05a36592283cfb3c01c5fee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_21f3713dccccb97b4eb1069dcfc4b772(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1fe904b24476c35715a7adb2ef77882e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2807962894439697], [2.2598795890808105], [1.870574712753296], [2.1764073371887207], [1.995255470275879], [2.09159779548645], [2.2656216621398926], [2.2172205448150635], [1.9983623027801514], [2.1377906799316406], [2.196044445037842], [2.0789554119110107], [2.1784939765930176], [2.2193682193756104], [2.052372694015503], [2.2238101959228516]], dtype='float32').reshape([16, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a57fc6f119ecc0fe65c415ba031270cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.960141658782959], [2.1533260345458984], [2.1658780574798584], [2.2659358978271484], [2.0110549926757812], [2.075639009475708], [2.0796830654144287], [1.8293005228042603], [2.01572847366333], [2.2285847663879395], [2.2727530002593994], [1.9443649053573608], [2.290123701095581], [1.808598279953003], [1.9616131782531738], [1.8629130125045776]], dtype='float32').reshape([16, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02f5afdd5f1c71dc093cf32d3f191456(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_75bab554da208b5f2ab8c20966062450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5bb9288f165109d85c8a56ee4482e8fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5bb9288f165109d85c8a56ee4482e8fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_74cbdc36bea1b892a68e6f6d72d198f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bd36ab8bffe19f223b731e7751d3012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74cbdc36bea1b892a68e6f6d72d198f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2b16151fc78307889e531ca3ccc821ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_09d3bdded2c6c3302a52fbd53c36189c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_aca2ca3628f4232b12a601106d0013d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_272eda8f533fec2346137bdee0096f63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_272eda8f533fec2346137bdee0096f63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8887f82a6181047e6ed7917ffff3c766(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b354d1f3df0f8dae957a12b27a6b5693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    
    class PrimitiveOp_e9e5134c925581257926065d90653a87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d03cb49337bdbc5ce78426307c003b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9e5134c925581257926065d90653a87
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_25ef81f1b8ab38a24842dcd704672d9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_fe4d39532a6cbba02657cad23c019eb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_709f13ece9c15cf2288967c98ac09510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_709f13ece9c15cf2288967c98ac09510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e5c4f921f153d881de4301b05dbaaf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e5c4f921f153d881de4301b05dbaaf85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_53b45a8f972246172423250f4a757bd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e9909792fda9f1c7a9eb6227caa277e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0ac7667ac231ee62d13e0755f5b7b996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0ac7667ac231ee62d13e0755f5b7b996(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_cd18fdfdc6782e57c98b9c33073c9b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cca7bd4e00c3de55be767f5b591f55fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_88e8065803d20969af7bc1d85c551bbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_0990f3e9d593369f9719a2e8cb9279a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_02d8e8634d589ef886572a306d2f228e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02d8e8634d589ef886572a306d2f228e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_72fe2a0ed966e67e5d3c86b16a94f591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_da0ec256c6781a9f626381b9df0defed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d2d7afcd84baaba31b51a769eedc3543(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b354d1f3df0f8dae957a12b27a6b5693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_c9c62a4e1329f848a9d5ec0c7c682588(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_66d553cc02838c6fda3a865d4bf83ed7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.2159955501556396], [2.2937707901000977], [1.9798015356063843], [1.930498719215393], [2.3208510875701904], [2.086238145828247], [2.125875949859619], [2.1606414318084717], [2.064732551574707], [2.2746167182922363], [1.9600178003311157], [1.9090816974639893], [2.188173294067383], [2.0657029151916504], [2.205883741378784], [2.1968834400177], [2.008652687072754], [1.8923864364624023], [2.230451822280884], [2.1479268074035645], [1.949690341949463], [1.9484350681304932], [2.20443058013916], [2.192775249481201]], dtype='float32').reshape([24, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dd16043acdc1f485e431d458781b06a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.1657958030700684], [2.0923051834106445], [2.312364339828491], [2.0151829719543457], [2.0399513244628906], [2.1100361347198486], [2.1869425773620605], [2.2116706371307373], [1.970133900642395], [1.8513662815093994], [2.2078945636749268], [2.220553159713745], [2.2786765098571777], [1.9406858682632446], [2.060137987136841], [1.9701740741729736], [2.385483741760254], [2.1726832389831543], [1.9692106246948242], [2.2667107582092285], [2.135495185852051], [2.080367088317871], [1.9795235395431519], [1.9408769607543945]], dtype='float32').reshape([24, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6288857815d5b5958abc37ec842bd731(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_b71261e1ea8995809ee9eae4d2fdeab7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f9f155b5d9dda4852e7c4ce3aa80ad1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f9f155b5d9dda4852e7c4ce3aa80ad1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_33fc27805959ed8f6a94eb2241e30581(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1595c487d28a66795f91d65dc57678a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ac8a5536937a5d309d2560360f205b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.888964295387268], [2.2693610191345215], [1.9160085916519165], [2.2557058334350586]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5239c5ec32d0c778d068295a14e8acbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[1.8958507776260376], [2.0913102626800537], [2.2948904037475586], [2.1589481830596924]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_06a1255461ce3547376c787ad9867a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74cbdc36bea1b892a68e6f6d72d198f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_22e6a75d89edc887a02e5cdce7dd6f64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74cbdc36bea1b892a68e6f6d72d198f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_39f5dabf3b5a3fb41cc76bd2f1d7ef4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_63151970a1fece2f0b525f32fd90c4e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_a38d01fccd98ea3f6e1e6d9128da19ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74cbdc36bea1b892a68e6f6d72d198f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_84966c751333f6371bb3022519a4a9d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_761a9854ac52bc32e92f377e7453ccdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ddbbe676006c6ce48ed7205cb09e4b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ddbbe676006c6ce48ed7205cb09e4b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_70920ee59459bd0790b2279d79152c68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b8b347e68cc22410fa44a8403ee42bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_54a13f76f5c30cbcb0af0134f0533bc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_72fe2a0ed966e67e5d3c86b16a94f591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_818c829da77ce5a15982335baa21d36c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f9b293b28a52f6dca9f842c520804f02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e161033fd33cc4e09fa993290d516a07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_767518980a9a3b02ce828c52d10bfcd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_767518980a9a3b02ce828c52d10bfcd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_83d904ffc70fd0011ff9ce8ea776119e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f2a9f54e2933b243e27f8da6edb1ca86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f2a9f54e2933b243e27f8da6edb1ca86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6a281e33529f818e5b265a26197f0b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9e5134c925581257926065d90653a87
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_62f0fb93a93c076e38b894a6e6ae07e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_f3737a89a15a5bd65f16905857f846f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f3353e229ce937f7a0f64631fda50b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3737a89a15a5bd65f16905857f846f8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7faabc287368c0d53196ff209ff3c1fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7faabc287368c0d53196ff209ff3c1fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e8904e5eda057c78884ae879a6d2876a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8a7ab988b74579214b7d4649a65dd41d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9e5134c925581257926065d90653a87
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_52e40e51faabe201371d2995aab0356f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74cbdc36bea1b892a68e6f6d72d198f9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0986fcdaaba0b504323522c525b05209(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.squeeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d8b49cd43d746d272e2e46692fb34cc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0986fcdaaba0b504323522c525b05209
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3a2831575e5fb50ed9777e3efd89fcf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a1ac346ed335b42a2cf35a98b5776ec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c1fc04a6793a277f876545db6190a741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_4e57091b79da69e9b6c79de03e69ed8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_793311bd82755f2f8fe802df18379192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ecd3edd87d10d72245bf7f7712eecb5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ecd3edd87d10d72245bf7f7712eecb5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9fc5e87150a2183149c5552b4508014f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9fc5e87150a2183149c5552b4508014f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5676cdc4466a218ae66b162d9cbe4e5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5676cdc4466a218ae66b162d9cbe4e5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8217c155953853fb0b7309f48ccfd34e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8217c155953853fb0b7309f48ccfd34e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1888e3bd8309ae6508d0bb1944102f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_1888e3bd8309ae6508d0bb1944102f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5dc895227b9451fef4fff8188545f4d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5dc895227b9451fef4fff8188545f4d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8f1b668b91ba5faada930d8c674a1350(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_ef946bcc28fb3fd23edcd981389098bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ef946bcc28fb3fd23edcd981389098bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da04d0dc235bf0709b90a29b33f0ed8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d59c47c0e503688e820620d77fa0a193(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_aca2ca3628f4232b12a601106d0013d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_30fb3b5088b7bb49ca79acc0c206f0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_7a7fc799924f84dea08c451c85bc41af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_9d935bf2c06c27434e7d97e581df971e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9e5134c925581257926065d90653a87
        def get_inputs(self):
            return [
                paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_69a362dbce56ea159309fe799c8043b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_cf6d268fa076e275205354a5b6a10e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_5cc88db370f2ba17cc4859214ea0ce99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.155951499938965], [1.864601731300354], [2.0482661724090576], [2.106358766555786], [2.1773722171783447], [2.072309970855713], [2.1956989765167236], [2.012533664703369], [2.229339122772217], [1.858726143836975], [2.2677783966064453], [2.1236958503723145], [2.0704946517944336], [2.188840389251709], [2.073134422302246], [2.157585620880127], [2.0383729934692383], [2.3608081340789795], [2.0037755966186523], [2.065575361251831]], dtype='float32').reshape([20, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e88d8c2bd372c369af165cfeb6c2dfca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81757b771fe54a1956d4f7c6dcec3480
        def get_inputs(self):
            return [
                paddle.to_tensor([[2.052689790725708], [2.2076961994171143], [2.0857365131378174], [1.846848964691162], [1.8959813117980957], [2.0812926292419434], [2.1710519790649414], [2.272707462310791], [2.07831072807312], [2.199514627456665], [2.1488406658172607], [2.2427282333374023], [2.193753480911255], [1.973362922668457], [2.1430397033691406], [2.0266079902648926], [2.1450860500335693], [2.1068367958068848], [2.3094241619110107], [1.8611677885055542]], dtype='float32').reshape([20, 1]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_818c829da77ce5a15982335baa21d36c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b8b347e68cc22410fa44a8403ee42bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_21f3713dccccb97b4eb1069dcfc4b772(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_20e1b3c3520b6dae437550b39fc198da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0aa1e1cd1e7d30ee055a1f5bca7410e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3737a89a15a5bd65f16905857f846f8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e161033fd33cc4e09fa993290d516a07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_753830c114eb8abff97895483ccc9890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_753830c114eb8abff97895483ccc9890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b50f83d4003fd3615ca02ccec8f1e897(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_70920ee59459bd0790b2279d79152c68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_500609b1203ae1baac72b17eb7727a14
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_eccc49266b10bf67c7385dc4fd53990f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e45072e21be0f0b041e382ab78bf2722
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_13e1d5b86b3b9459c67741f4aeddd79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_e9f54a2a08b4dd9f2c5f2d8b863fb37f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e9f54a2a08b4dd9f2c5f2d8b863fb37f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e1a4b6db890293550e00602dbfc3b621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_127150ed89d10ae71181882049044e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_1f49bffef6d54f7250041daa3cce303b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0986fcdaaba0b504323522c525b05209
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_246cbf611716b22079cf2d29012d62bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9bb0689d6fcb3daadf9d76710630315
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()