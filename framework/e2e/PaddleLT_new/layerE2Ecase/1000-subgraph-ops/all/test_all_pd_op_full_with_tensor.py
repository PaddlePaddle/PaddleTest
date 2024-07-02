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
    class PrimitiveOp_60085b9a8f52ae8cb236832138dc9a00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1024, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c9612678371c2a748a639ae91dd925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60085b9a8f52ae8cb236832138dc9a00
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_a118a10d557c2627362778e4e43868c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 4096, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d283b38ae4baab5a5714874744b02b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a118a10d557c2627362778e4e43868c5
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b3e36aeb21358c91c3313b4271a6e30c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 16384, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_384732d7806a17d48b0a0f4619a69ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3e36aeb21358c91c3313b4271a6e30c
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1c9612678371c2a748a639ae91dd925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60085b9a8f52ae8cb236832138dc9a00
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9d283b38ae4baab5a5714874744b02b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a118a10d557c2627362778e4e43868c5
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_384732d7806a17d48b0a0f4619a69ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3e36aeb21358c91c3313b4271a6e30c
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_384732d7806a17d48b0a0f4619a69ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3e36aeb21358c91c3313b4271a6e30c
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9d283b38ae4baab5a5714874744b02b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a118a10d557c2627362778e4e43868c5
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1c9612678371c2a748a639ae91dd925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60085b9a8f52ae8cb236832138dc9a00
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_2730ce25ac64eaa07e7732ed8a678703(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 256, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2101ca34e484950276385386c8f665ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2730ce25ac64eaa07e7732ed8a678703
        def get_inputs(self):
            return [
                paddle.to_tensor([64.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_b095cccb05bdf40cfb40721d9c067828(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 64, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6c8f98e36b1dded71db65f5892f3d47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b095cccb05bdf40cfb40721d9c067828
        def get_inputs(self):
            return [
                paddle.to_tensor([128.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1c9612678371c2a748a639ae91dd925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60085b9a8f52ae8cb236832138dc9a00
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9d283b38ae4baab5a5714874744b02b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a118a10d557c2627362778e4e43868c5
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_384732d7806a17d48b0a0f4619a69ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3e36aeb21358c91c3313b4271a6e30c
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1c9612678371c2a748a639ae91dd925f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60085b9a8f52ae8cb236832138dc9a00
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9d283b38ae4baab5a5714874744b02b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a118a10d557c2627362778e4e43868c5
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_384732d7806a17d48b0a0f4619a69ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3e36aeb21358c91c3313b4271a6e30c
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8f5e2043f939d8be90ff64f53b0730ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1024, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae9223533a80dd076e88c75d749b833e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f5e2043f939d8be90ff64f53b0730ad
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ed873db6832ab3de415af911b709d5da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 4096, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7e96802d1e6b138bcc37ff42e8dcc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed873db6832ab3de415af911b709d5da
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_f5c5e167809d8df19ee56a2bab077e72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 16384, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a04f5ade538ea9f4f4a35effcdac5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5c5e167809d8df19ee56a2bab077e72
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ae9223533a80dd076e88c75d749b833e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f5e2043f939d8be90ff64f53b0730ad
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7e96802d1e6b138bcc37ff42e8dcc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed873db6832ab3de415af911b709d5da
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0a04f5ade538ea9f4f4a35effcdac5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5c5e167809d8df19ee56a2bab077e72
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0a04f5ade538ea9f4f4a35effcdac5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5c5e167809d8df19ee56a2bab077e72
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7e96802d1e6b138bcc37ff42e8dcc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed873db6832ab3de415af911b709d5da
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ae9223533a80dd076e88c75d749b833e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f5e2043f939d8be90ff64f53b0730ad
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_6f3d849e42eacf03fad3f314052cb7c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 256, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61b64a85491dcdec34ff43eb2e9bdd58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f3d849e42eacf03fad3f314052cb7c3
        def get_inputs(self):
            return [
                paddle.to_tensor([64.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ac631753a7ba7c93fdfb9a6bbbde0599(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 64, 1]
            return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c0a5dcf7dd941787668e727fb0bf1c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac631753a7ba7c93fdfb9a6bbbde0599
        def get_inputs(self):
            return [
                paddle.to_tensor([128.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ae9223533a80dd076e88c75d749b833e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f5e2043f939d8be90ff64f53b0730ad
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7e96802d1e6b138bcc37ff42e8dcc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed873db6832ab3de415af911b709d5da
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0a04f5ade538ea9f4f4a35effcdac5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5c5e167809d8df19ee56a2bab077e72
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ae9223533a80dd076e88c75d749b833e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f5e2043f939d8be90ff64f53b0730ad
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7e96802d1e6b138bcc37ff42e8dcc50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed873db6832ab3de415af911b709d5da
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0a04f5ade538ea9f4f4a35effcdac5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5c5e167809d8df19ee56a2bab077e72
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0], dtype='float32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()