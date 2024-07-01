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
    class PrimitiveOp_297be2ff90959e60068ef7ea422bc711(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b63853834a0ec9d712d2165d6ccd979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_699da104a6c4922c037298fc787b67b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_487ba46effd6cbdc7628bdbb35f34964(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8822cdda1c6da8c5341bd5ff4b296bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_487ba46effd6cbdc7628bdbb35f34964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a4fdbaf5723cd84b79827737199d095(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.pow(input_0, 3)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fde1a8de16d6870c7c3987c4b5376b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4fdbaf5723cd84b79827737199d095
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1643868088722229], [0.05441196262836456], [0.22459563612937927], [0.4925847351551056], [0.35154488682746887], [0.19777941703796387]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_8822cdda1c6da8c5341bd5ff4b296bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_487ba46effd6cbdc7628bdbb35f34964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f73f3d88d4800eef77aba07c71fe4327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4fdbaf5723cd84b79827737199d095
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.10589680820703506], [0.2850358188152313], [0.26924020051956177], [0.16354554891586304], [0.1850755661725998], [0.1845693290233612]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_200a04346391952f1d7ac664cda308cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_04c9e5b5c2e8f6304a22d8ab8236a5c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_021b72446d9386bc7418d8e6d80d63b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04c9e5b5c2e8f6304a22d8ab8236a5c4
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ff0df962f6510e16db80ff86833bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_487ba46effd6cbdc7628bdbb35f34964
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b0893cd2caa8852bb1dee3fc14ec17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04c9e5b5c2e8f6304a22d8ab8236a5c4
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a1d33fabbc5b2fdad6876cf40a2b5da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_601df6a99b993d53639dc7a065923f2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f8fe9f2ffcc521cc34ba3651e054f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_706af293c5faeb63a0072844a3d37449(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04c9e5b5c2e8f6304a22d8ab8236a5c4
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f769f2e03c97067864136798c8e018b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49892520d9a7ea1189dc265a2a2f2628(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bee1b894bafe9bea6b571a4a6caff5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49d2dd34e8511e07b8136a8f2fc69f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebc5515ebb907c3008a3b4ace3780750(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04c9e5b5c2e8f6304a22d8ab8236a5c4
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b63853834a0ec9d712d2165d6ccd979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0315e5c5edccbd218477c1f1659b832e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04c9e5b5c2e8f6304a22d8ab8236a5c4
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4c6eefe2be062b2a1233d5f794f3822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_297be2ff90959e60068ef7ea422bc711
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.pow(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3795c92d40bf5f3e000e2b7d5a57029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8133d0d05efdcea47df96ae86d7398b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8822cdda1c6da8c5341bd5ff4b296bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_487ba46effd6cbdc7628bdbb35f34964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fde1a8de16d6870c7c3987c4b5376b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4fdbaf5723cd84b79827737199d095
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1643868088722229], [0.05441196262836456], [0.22459563612937927], [0.4925847351551056], [0.35154488682746887], [0.19777941703796387]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_8822cdda1c6da8c5341bd5ff4b296bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_487ba46effd6cbdc7628bdbb35f34964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f73f3d88d4800eef77aba07c71fe4327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a4fdbaf5723cd84b79827737199d095
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.10589680820703506], [0.2850358188152313], [0.26924020051956177], [0.16354554891586304], [0.1850755661725998], [0.1845693290233612]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_6d18854d470643016f359821a0701dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d3242b85a4fcf22da7e7d3783926a7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ff0df962f6510e16db80ff86833bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_487ba46effd6cbdc7628bdbb35f34964
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e6e3f020421e91a085eafe21e3c632c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c57a383e8404e13ae8310411953f7773(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0be36e3033d7459afbed520d9e63528d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9246033eae8642adc4b1b62e53587328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f360bb9fee19248c3676911ce138a596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53f0cdb5afd15cccfb563ea2303d7a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33a816cbeb299f63c6ae3b85aae93c88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_657e2f82e3e3e4f7e91b2bb2b242ce66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a893f1b64a37ae74f48201dfea2c8de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b947a191d157af7fbb003630bf58adbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a41dd6a9d9f2716970a1128f459b4f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f64a816b749569bc8a2d04e9f4034a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b525d6949cc89f8bedcdb4d4d31d0204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34da5f131085f211a91d3248d94d9c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b3630d38e9c369d017173ef782b2ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb969b41b704bf7bfcd9e7d0041b7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_601df6a99b993d53639dc7a065923f2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac2ea33ef18e4317d2d5af6639509ef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3795c92d40bf5f3e000e2b7d5a57029f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_498992c08dc93116782c4f5b76257b4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72bb1c2a9915717699879e04559ae497(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f95d5a0517ca64a5d807c5b5c882d42
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()