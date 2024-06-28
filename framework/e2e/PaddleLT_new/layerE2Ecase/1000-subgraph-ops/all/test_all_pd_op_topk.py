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
    class PrimitiveOp_971d87c8247ca21a7868036a8fe63784(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83abf5863f18f48102c33bcb163430c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_971d87c8247ca21a7868036a8fe63784
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b65881100295762c598ef1fb017cbd9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab8fdde1e1f69c8e98231afa2540f0bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b65881100295762c598ef1fb017cbd9b
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4cd98994c8739970f3c3117247ae5158(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7183332cc180e90972803388df18b1b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cd98994c8739970f3c3117247ae5158
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f223c7b2698b838974634cc92f7ff282(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_207e821593009fb38683eac75497d117(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f223c7b2698b838974634cc92f7ff282
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_edf4ff481674f5c8355247d4ba31f7fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47ee6ec082e758de86f497cddf799040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edf4ff481674f5c8355247d4ba31f7fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e83e4aba60cde372738b3ffe2d6f2dd3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c16fae7dc2c63f0e28b2e0ecc377be6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e83e4aba60cde372738b3ffe2d6f2dd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0d0dbaab1d1edc4f2d24b39671813055(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_688453c4efb86b7f8cfc6b93c367db9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d0dbaab1d1edc4f2d24b39671813055
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_886986b741428b89295d135a8a5e9eff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01d34f999d3ad3f2af1111093a3e470d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_886986b741428b89295d135a8a5e9eff
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e4aba9124d03db4c468a8469b8796dac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02b10f0d303f7978ffd1c9e183692a88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4aba9124d03db4c468a8469b8796dac
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2e52ea1506fe17d75897c59e5391c992(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 17, 100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d355cc7c06d0cdce2ad98a51585c27e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e52ea1506fe17d75897c59e5391c992
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b53cf69b8d535ad3ea456f6e7e5e260e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.topk(input_0, input_1, 2, True, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89ea1e5839cbfc3312ff875e57a908ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b53cf69b8d535ad3ea456f6e7e5e260e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_eadfc4a1147d3d993aadc11d519de04b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b53cf69b8d535ad3ea456f6e7e5e260e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_03c96ec7646652d16b7f02cc2ec7df5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b53cf69b8d535ad3ea456f6e7e5e260e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_68c4f4c37675a19b3b1a03467245e410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b53cf69b8d535ad3ea456f6e7e5e260e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a8f4f7f07cbbd65f4febc1a49a6f6a07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b53cf69b8d535ad3ea456f6e7e5e260e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()