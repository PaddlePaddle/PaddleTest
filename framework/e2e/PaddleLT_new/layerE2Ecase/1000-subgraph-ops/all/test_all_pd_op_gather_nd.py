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
    class PrimitiveOp_993a4424a34b98093a203a8c20dfb9e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e4fd9fcf18e7c0205d1032e3da1b0c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_993a4424a34b98093a203a8c20dfb9e9
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3136, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_65143d3167fb2d0ea1832367deee26a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 500, 2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_badbcf3cc8777676428bb3c12d574dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65143d3167fb2d0ea1832367deee26a8
        def get_inputs(self):
            return [
                paddle.uniform([1, 41344, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_ea568d839cad74ad0edd219d9386d54c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65143d3167fb2d0ea1832367deee26a8
        def get_inputs(self):
            return [
                paddle.uniform([1, 25920, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_9047ea8c1f14d94f7a4b8c5254cb48c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_993a4424a34b98093a203a8c20dfb9e9
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2352, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_426943de289fa12c5f312affd421638e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_993a4424a34b98093a203a8c20dfb9e9
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4704, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_bc87355468ad8edf2b57ab749d8306b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_993a4424a34b98093a203a8c20dfb9e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1568, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3209e5acb84bdf12fb439d1a02a0085c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65143d3167fb2d0ea1832367deee26a8
        def get_inputs(self):
            return [
                paddle.uniform([1, 11520, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    
    class PrimitiveOp_bac938114963fb5ecd2e0cb5ac825178(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[3136, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47a7589bdc8942e33dcc94bff145c15d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bac938114963fb5ecd2e0cb5ac825178
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3136, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_27d04e3722e24485ab76e7c4a13bedc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 41344, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 500, 2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c13e2efeb9fd9a72670d76cf4d42333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27d04e3722e24485ab76e7c4a13bedc0
        def get_inputs(self):
            return [
                paddle.uniform([1, 41344, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    
    class PrimitiveOp_3a862e22c6d6e6aa0b74c2adc32176bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 25920, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 500, 2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb9dd9cdf31b39ab244a2768c299a57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a862e22c6d6e6aa0b74c2adc32176bd
        def get_inputs(self):
            return [
                paddle.uniform([1, 25920, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    
    class PrimitiveOp_01e5ede916b5808a1e225e2785bd1c26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[2352, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fac6aab60a18c2dc17b69e7ca83545d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01e5ede916b5808a1e225e2785bd1c26
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2352, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_d3d2ae138b189c119b2659648765bf2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[4704, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3530a0248e7ffc5c6a99980e6dce962(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3d2ae138b189c119b2659648765bf2b
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4704, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_4a7b1655a63811d27535e495a50ce091(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[1568, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd23029a4524dcb9b6ad91a974153440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a7b1655a63811d27535e495a50ce091
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1568, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_9f1fd5c5dac682ad11cc56b97bb8ed73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11520, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 500, 2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_247235eb142da6d56360b78c74c14f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f1fd5c5dac682ad11cc56b97bb8ed73
        def get_inputs(self):
            return [
                paddle.uniform([1, 11520, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    
    class PrimitiveOp_66e392720f75929134218a808ef28832(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f62a05d0e2098ed346df9566de89d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66e392720f75929134218a808ef28832
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3136, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_2fafbd10392a0c5d6bdf8bb9140d06e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61d7398be02889d087e6a66d3abd47dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fafbd10392a0c5d6bdf8bb9140d06e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 41344, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_c38ffbee2ee2c6b675c74a24dc069061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fafbd10392a0c5d6bdf8bb9140d06e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 25920, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_c63bb5916599f4dd73476058c2185acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66e392720f75929134218a808ef28832
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2352, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ea8bad011b15fe6ae7ddea8fe05fc471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66e392720f75929134218a808ef28832
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4704, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_22a94a7aac2cb8fd7eeac02af0b558bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66e392720f75929134218a808ef28832
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1568, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_37bf0afaf49bf67aa62bb10fad817f23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fafbd10392a0c5d6bdf8bb9140d06e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 11520, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    

if __name__ == '__main__':
    unittest.main()