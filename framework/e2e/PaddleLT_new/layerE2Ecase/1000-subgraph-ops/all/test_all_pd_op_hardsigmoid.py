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
            PADDLE_DEBUG_ENABLE_CINN=False,
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
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
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





last_stage_failed = (IsCinnStageEnableDiff() and LastCINNStageFailed())
class PrimitiveOp_b9284eb8f426882f68ce0c1dde964d96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea6d94a9d6c0d217e3c8a961ccefe943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9284eb8f426882f68ce0c1dde964d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1661ddab006fce894154dbe8787549e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_244b21221fc6c87ae3389a62861279c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1661ddab006fce894154dbe8787549e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_43508499af60ea1295741df9568d3735(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1e76b67656cd0ffdfa61e79ee79651ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17c2c9c12d0a1a839ddee512ba974b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e76b67656cd0ffdfa61e79ee79651ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3f80510d48a914b06cff260da230cbc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3b834d3a004eca5b8590c3918c7de1a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86593428f1ccbbe1ab1472ae89fb8532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b834d3a004eca5b8590c3918c7de1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f1ed7b0104f9089010fea9e3895c0ecb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a44a4a59aa9dab0a419e05527a8cc736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1ed7b0104f9089010fea9e3895c0ecb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7a6303b006e27249cc819d683be4a633(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7404a334ae4601520888c13ffb76b5cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ff99f82a62319c07c528b3119c1ea10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7404a334ae4601520888c13ffb76b5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4f5fde49ec09f44e999af07440637f4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c609797713542e8389d03a7d72fadea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d94cc4012f13417abf1ef45446006cdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7e77886d8248411172d3af7d1ead59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d94cc4012f13417abf1ef45446006cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ff99f82a62319c07c528b3119c1ea10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7404a334ae4601520888c13ffb76b5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_84b407ee620a688d42e51dab19e38b36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5742cb8e1bb761ff95da5bab5213b68d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e8b537a037b813ded2c5575b81ef0c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5742cb8e1bb761ff95da5bab5213b68d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3080856800079346]], [[1.3405048847198486]], [[1.0662840604782104]], [[1.062343716621399]], [[0.7937247157096863]], [[1.1227585077285767]], [[1.5744913816452026]], [[0.8420699834823608]], [[0.8087244033813477]], [[0.40220385789871216]], [[1.434902310371399]], [[1.0336328744888306]], [[0.934656023979187]], [[1.1584774255752563]], [[1.4022589921951294]], [[1.7274384498596191]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ac7f78858d2084f5e1ce28668c8a840c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d34be0176b9d92fcd19edfe03006fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac7f78858d2084f5e1ce28668c8a840c
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ff99f82a62319c07c528b3119c1ea10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7404a334ae4601520888c13ffb76b5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea6d94a9d6c0d217e3c8a961ccefe943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9284eb8f426882f68ce0c1dde964d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9846433537798d518a07eb3e040b54c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6dc4bf8805943b22d4193921f51fecbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9846433537798d518a07eb3e040b54c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ae1d1c912aaefbeb310efa1c5baaba48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49c05104ebe1161ed7a0602587ff35de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae1d1c912aaefbeb310efa1c5baaba48
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea6d94a9d6c0d217e3c8a961ccefe943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9284eb8f426882f68ce0c1dde964d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a44a4a59aa9dab0a419e05527a8cc736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1ed7b0104f9089010fea9e3895c0ecb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5794c631cd125e6b7aea1ca202167585(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1c9aa211f1345698d2e7eab6d26f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5794c631cd125e6b7aea1ca202167585
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_66944b978dca3fd6e8f8f838776add6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4ffb6da900944042e6bdbf236b37856e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66944b978dca3fd6e8f8f838776add6d
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ff99f82a62319c07c528b3119c1ea10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7404a334ae4601520888c13ffb76b5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bf392cf805ac592db83c562513386d02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37f49030bacd1560fcbbdaa30c9ef746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf392cf805ac592db83c562513386d02
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e86f7c75c82e20a0dd23b342e595579e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab97b538d6e58fd80f3ebab8c762dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86f7c75c82e20a0dd23b342e595579e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_516d6e36be8e85708c0dbe570e542565(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af7f7fce803a3d6b7fe3e3c4daadd54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_516d6e36be8e85708c0dbe570e542565
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1c9aa211f1345698d2e7eab6d26f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5794c631cd125e6b7aea1ca202167585
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86593428f1ccbbe1ab1472ae89fb8532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b834d3a004eca5b8590c3918c7de1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c29a96e9cc8a6d23e916f7bed81286b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b500e1178ce3b608bb743fff217eb79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c29a96e9cc8a6d23e916f7bed81286b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17c2c9c12d0a1a839ddee512ba974b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e76b67656cd0ffdfa61e79ee79651ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_244b21221fc6c87ae3389a62861279c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1661ddab006fce894154dbe8787549e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b500e1178ce3b608bb743fff217eb79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c29a96e9cc8a6d23e916f7bed81286b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab97b538d6e58fd80f3ebab8c762dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86f7c75c82e20a0dd23b342e595579e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7e77886d8248411172d3af7d1ead59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d94cc4012f13417abf1ef45446006cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86593428f1ccbbe1ab1472ae89fb8532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b834d3a004eca5b8590c3918c7de1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7e77886d8248411172d3af7d1ead59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d94cc4012f13417abf1ef45446006cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a44a4a59aa9dab0a419e05527a8cc736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1ed7b0104f9089010fea9e3895c0ecb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49c05104ebe1161ed7a0602587ff35de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae1d1c912aaefbeb310efa1c5baaba48
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7e94cea61f75e285f382bf264cf2a7c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d91a85d87947146982ae7120871b375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e94cea61f75e285f382bf264cf2a7c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17c2c9c12d0a1a839ddee512ba974b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e76b67656cd0ffdfa61e79ee79651ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_08dad864966eeddfd5fef09a95724d86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0abb730f3d5ccfb93de91c226bc62318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08dad864966eeddfd5fef09a95724d86
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7e77886d8248411172d3af7d1ead59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d94cc4012f13417abf1ef45446006cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_fa7acd264d34f8435ca6f9517456f4b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9bdaf71deacccfd4ade504a34e3809a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa7acd264d34f8435ca6f9517456f4b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_72cd7ac71c5fc4833c452364dbf3e008(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c50b06cda92ff8a63abc24afb84da05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72cd7ac71c5fc4833c452364dbf3e008
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7e77886d8248411172d3af7d1ead59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d94cc4012f13417abf1ef45446006cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2607a8cea70eb187004758bf5e86c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5742cb8e1bb761ff95da5bab5213b68d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5150817632675171]], [[1.4809907674789429]], [[0.7190375328063965]], [[1.238405466079712]], [[1.6407709121704102]], [[1.2735278606414795]], [[1.206140398979187]], [[1.5463407039642334]], [[0.9320278167724609]], [[1.4607892036437988]], [[1.585608720779419]], [[1.102509617805481]], [[1.5226378440856934]], [[0.9293451309204102]], [[1.0666732788085938]], [[1.6668328046798706]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9bdaf71deacccfd4ade504a34e3809a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa7acd264d34f8435ca6f9517456f4b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37f49030bacd1560fcbbdaa30c9ef746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf392cf805ac592db83c562513386d02
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e08749971ddfd92c336276f074828b1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da74b8a2e6eeffadfc27ab7a4f183f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08749971ddfd92c336276f074828b1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab97b538d6e58fd80f3ebab8c762dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86f7c75c82e20a0dd23b342e595579e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c9aecca9d3e968ea573fed8ef2545670(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ee2cf0dcc9bc7f207f8ef48d3e407f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9aecca9d3e968ea573fed8ef2545670
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea6d94a9d6c0d217e3c8a961ccefe943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9284eb8f426882f68ce0c1dde964d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af7f7fce803a3d6b7fe3e3c4daadd54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_516d6e36be8e85708c0dbe570e542565
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c3ee1c5dca9805c3085b07fa2d207914(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9156336d2968d280b2f7fd4a0e7dfc1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ee1c5dca9805c3085b07fa2d207914
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af7f7fce803a3d6b7fe3e3c4daadd54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_516d6e36be8e85708c0dbe570e542565
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d34be0176b9d92fcd19edfe03006fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac7f78858d2084f5e1ce28668c8a840c
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da74b8a2e6eeffadfc27ab7a4f183f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08749971ddfd92c336276f074828b1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_25af58f266101c97770dc8ff8158e313(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80b624e5b1e026ef9c8cda44e723860d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af58f266101c97770dc8ff8158e313
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea6d94a9d6c0d217e3c8a961ccefe943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9284eb8f426882f68ce0c1dde964d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d91a85d87947146982ae7120871b375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e94cea61f75e285f382bf264cf2a7c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af7f7fce803a3d6b7fe3e3c4daadd54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_516d6e36be8e85708c0dbe570e542565
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e4570dca7b6755147ad363f0cb71553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e4570dca7b6755147ad363f0cb71553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e4570dca7b6755147ad363f0cb71553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0e4570dca7b6755147ad363f0cb71553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d74c4b0f17a64b3cfe141c5f7777ee94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d74c4b0f17a64b3cfe141c5f7777ee94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d74c4b0f17a64b3cfe141c5f7777ee94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d74c4b0f17a64b3cfe141c5f7777ee94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_eaec998794e0e4d4365e2d5f503c9325(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_037bd2fafa6aac77b3ac5b216858ddb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaec998794e0e4d4365e2d5f503c9325
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9bdaf71deacccfd4ade504a34e3809a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa7acd264d34f8435ca6f9517456f4b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4ffb6da900944042e6bdbf236b37856e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66944b978dca3fd6e8f8f838776add6d
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab97b538d6e58fd80f3ebab8c762dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86f7c75c82e20a0dd23b342e595579e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_99f992daf313e114803ee796dd4eef1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c69d46946578c3cf1931c02681d793ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99f992daf313e114803ee796dd4eef1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17c2c9c12d0a1a839ddee512ba974b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e76b67656cd0ffdfa61e79ee79651ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0abb730f3d5ccfb93de91c226bc62318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08dad864966eeddfd5fef09a95724d86
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86593428f1ccbbe1ab1472ae89fb8532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b834d3a004eca5b8590c3918c7de1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b17c2c9c12d0a1a839ddee512ba974b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e76b67656cd0ffdfa61e79ee79651ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_037bd2fafa6aac77b3ac5b216858ddb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaec998794e0e4d4365e2d5f503c9325
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1c9aa211f1345698d2e7eab6d26f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5794c631cd125e6b7aea1ca202167585
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_39d72285f44c590f8864e1591edfb8a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b6af414caaf65a5737b80d48a43d997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39d72285f44c590f8864e1591edfb8a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c7e77886d8248411172d3af7d1ead59c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d94cc4012f13417abf1ef45446006cdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0abb730f3d5ccfb93de91c226bc62318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08dad864966eeddfd5fef09a95724d86
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab97b538d6e58fd80f3ebab8c762dba6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e86f7c75c82e20a0dd23b342e595579e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6dc4bf8805943b22d4193921f51fecbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9846433537798d518a07eb3e040b54c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49c05104ebe1161ed7a0602587ff35de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae1d1c912aaefbeb310efa1c5baaba48
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_22fe74d17414a5fe61c8d52d689d1f87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aea125f4d1ee0b94080e5870e3802992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22fe74d17414a5fe61c8d52d689d1f87
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9156336d2968d280b2f7fd4a0e7dfc1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ee1c5dca9805c3085b07fa2d207914
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d91a85d87947146982ae7120871b375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e94cea61f75e285f382bf264cf2a7c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5f515c81e07a2a225d0ac1123a9b7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a6303b006e27249cc819d683be4a633
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80b624e5b1e026ef9c8cda44e723860d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25af58f266101c97770dc8ff8158e313
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b500e1178ce3b608bb743fff217eb79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c29a96e9cc8a6d23e916f7bed81286b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3af7f7fce803a3d6b7fe3e3c4daadd54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_516d6e36be8e85708c0dbe570e542565
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1c9aa211f1345698d2e7eab6d26f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5794c631cd125e6b7aea1ca202167585
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0568fae305dfd03d7a299ad95a792042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f80510d48a914b06cff260da230cbc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ff99f82a62319c07c528b3119c1ea10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7404a334ae4601520888c13ffb76b5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aea125f4d1ee0b94080e5870e3802992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22fe74d17414a5fe61c8d52d689d1f87
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86593428f1ccbbe1ab1472ae89fb8532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b834d3a004eca5b8590c3918c7de1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ee2cf0dcc9bc7f207f8ef48d3e407f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9aecca9d3e968ea573fed8ef2545670
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8159fbf19355ff459eb8f4c946cb0f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dfef85276ef7c7bd612221d2f6038e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a44a4a59aa9dab0a419e05527a8cc736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1ed7b0104f9089010fea9e3895c0ecb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_037bd2fafa6aac77b3ac5b216858ddb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaec998794e0e4d4365e2d5f503c9325
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da74b8a2e6eeffadfc27ab7a4f183f03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e08749971ddfd92c336276f074828b1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4ffb6da900944042e6bdbf236b37856e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66944b978dca3fd6e8f8f838776add6d
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_372be5f534693e0599d11eca7dc6688f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ff0830cf627a4c9702a73770b19e0a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1c9aa211f1345698d2e7eab6d26f030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5794c631cd125e6b7aea1ca202167585
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49c05104ebe1161ed7a0602587ff35de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae1d1c912aaefbeb310efa1c5baaba48
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ce42670be183ee0a7b19a1ac4a6fd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43508499af60ea1295741df9568d3735
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec790d8cc999a0ef1c63b68e4d7715c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6969439e8ffbd3d03eccf7b4be79647
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40c013375dd07f6eaac33dc70554954d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84b407ee620a688d42e51dab19e38b36
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c609797713542e8389d03a7d72fadea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c609797713542e8389d03a7d72fadea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c609797713542e8389d03a7d72fadea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c609797713542e8389d03a7d72fadea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_705f3fd77c047148f272d5273312f92d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40394.7109375]], [[41053.18359375]], [[58203.19140625]], [[47845.3359375]], [[44826.7734375]], [[30734.732421875]], [[48187.421875]], [[27655.919921875]], [[62914.5234375]], [[37999.03515625]], [[43159.25390625]], [[48522.34375]], [[56747.484375]], [[47576.82421875]], [[71490.4375]], [[41356.09765625]], [[27665.529296875]], [[51633.05078125]], [[35305.46875]], [[64955.77734375]], [[49642.28125]], [[45967.33203125]], [[43274.34765625]], [[52577.59375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9853426a004414cc94136830082d787d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32447.328125]], [[51521.078125]], [[55027.69921875]], [[56319.2109375]], [[51922.15625]], [[48465.328125]], [[44730.81640625]], [[44645.2265625]], [[36616.3671875]], [[60146.33984375]], [[75195.3828125]], [[49811.3515625]], [[69967.984375]], [[20346.373046875]], [[55820.28125]], [[50153.73046875]], [[55559.390625]], [[53131.55859375]], [[73189.2265625]], [[32012.06640625]], [[58937.90625]], [[50805.03125]], [[67007.3203125]], [[65413.1640625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80d17fd89a43ca47a8d59c2aa037f03c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[70237.3203125]], [[51783.8203125]], [[41368.76171875]], [[36558.3828125]], [[74452.875]], [[41336.609375]], [[66183.75]], [[90987.4453125]], [[65005.82421875]], [[70354.6875]], [[80621.34375]], [[61925.55078125]], [[71502.203125]], [[58610.109375]], [[56892.04296875]], [[61840.93359375]], [[87014.0703125]], [[43695.94140625]], [[55268.40625]], [[51172.734375]], [[86102.890625]], [[47229.51953125]], [[55730.66796875]], [[67458.921875]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_918f9f94e6ee6fb5d06181d0011ba5c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a95c2d46e514649ec0eee7700cd37d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[68354.5234375]], [[57062.53125]], [[33374.52734375]], [[49705.4765625]], [[74635.0703125]], [[87549.1640625]], [[47589.55078125]], [[42842.4140625]], [[53218.18359375]], [[58559.18359375]], [[57287.1640625]], [[34865.0546875]], [[43199.37890625]], [[70856.046875]], [[54579.44140625]], [[39887.65625]], [[48813.6328125]], [[103218.6953125]], [[67022.3984375]], [[28652.015625]], [[82444.8125]], [[52055.05859375]], [[46388.23046875]], [[92584.9921875]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ff99f82a62319c07c528b3119c1ea10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7404a334ae4601520888c13ffb76b5cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c69d46946578c3cf1931c02681d793ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99f992daf313e114803ee796dd4eef1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c609797713542e8389d03a7d72fadea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5fde49ec09f44e999af07440637f4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86593428f1ccbbe1ab1472ae89fb8532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b834d3a004eca5b8590c3918c7de1a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60e4e6b924eb5506b575d42834678c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_881e3930ee3c28d837ad998b06cc1e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.166667, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e61480af4ab77b3c7733116480b40a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, 0.2, 0.5)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbbfc959ec9fe4186e94c170e3fcc8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc48eff68ec4077c8e1f2eed586f98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03970d371695ea096e5ec50f09a7338c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38453794e705fcacbd4f772ce2be8a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a392b3b769f9db6c5f71bd23c8328ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9124d93e2163bc5e9bc29a98bf0779f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14e7447ff50bec88a027dc4e6e570e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a392b3b769f9db6c5f71bd23c8328ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91dab0f856e1c1a72158ee5cbb187ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3080856800079346]], [[1.3405048847198486]], [[1.0662840604782104]], [[1.062343716621399]], [[0.7937247157096863]], [[1.1227585077285767]], [[1.5744913816452026]], [[0.8420699834823608]], [[0.8087244033813477]], [[0.40220385789871216]], [[1.434902310371399]], [[1.0336328744888306]], [[0.934656023979187]], [[1.1584774255752563]], [[1.4022589921951294]], [[1.7274384498596191]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe112e3f2de935c44d71553954ccb220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a392b3b769f9db6c5f71bd23c8328ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e61480af4ab77b3c7733116480b40a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ab49625ba9269bc081d59d08df1ff6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43dec9069fcf5af6cd70752fdeaa573b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e61480af4ab77b3c7733116480b40a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38453794e705fcacbd4f772ce2be8a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fc0ca9e6ba14ea7124412e79343faa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58b88480a6a3a250a2eb80994a0049e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a392b3b769f9db6c5f71bd23c8328ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe3ca5d2c6c7edc9ba4c791bf69b165c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe1fe5417f9b51083fb847e9d229b827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8a6312b07a3968832944def70532a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fc0ca9e6ba14ea7124412e79343faa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03970d371695ea096e5ec50f09a7338c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5291f68250bd92253d89e06c571a0bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc48eff68ec4077c8e1f2eed586f98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbbfc959ec9fe4186e94c170e3fcc8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5291f68250bd92253d89e06c571a0bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe1fe5417f9b51083fb847e9d229b827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14e7447ff50bec88a027dc4e6e570e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03970d371695ea096e5ec50f09a7338c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14e7447ff50bec88a027dc4e6e570e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38453794e705fcacbd4f772ce2be8a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43dec9069fcf5af6cd70752fdeaa573b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c4a10731c9c33703de56afbb658fe27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc48eff68ec4077c8e1f2eed586f98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c6761d28fb6701fb0a16d9cf937890c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14e7447ff50bec88a027dc4e6e570e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8653a5fc28d4605b65c2035b2e29717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3d9bc308e2100bdb1f9fc0316811e525(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14e7447ff50bec88a027dc4e6e570e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41800566fb559967f2dd5a827611e49f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5150817632675171]], [[1.4809907674789429]], [[0.7190375328063965]], [[1.238405466079712]], [[1.6407709121704102]], [[1.2735278606414795]], [[1.206140398979187]], [[1.5463407039642334]], [[0.9320278167724609]], [[1.4607892036437988]], [[1.585608720779419]], [[1.102509617805481]], [[1.5226378440856934]], [[0.9293451309204102]], [[1.0666732788085938]], [[1.6668328046798706]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8653a5fc28d4605b65c2035b2e29717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe3ca5d2c6c7edc9ba4c791bf69b165c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5deda1a8809f68c6d9fbe517805c336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe1fe5417f9b51083fb847e9d229b827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5635388277fd20b53d2faf50afbe905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e61480af4ab77b3c7733116480b40a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8a6312b07a3968832944def70532a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_548e70a986ddfdd6e59284e665df51c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8a6312b07a3968832944def70532a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe112e3f2de935c44d71553954ccb220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5deda1a8809f68c6d9fbe517805c336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b983b1cc73b530e06ae8f5541f642b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e61480af4ab77b3c7733116480b40a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c4a10731c9c33703de56afbb658fe27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8a6312b07a3968832944def70532a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45c878442af40ca58a8ab0432ad14256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45c878442af40ca58a8ab0432ad14256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45c878442af40ca58a8ab0432ad14256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45c878442af40ca58a8ab0432ad14256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95c00c3d3a8f216d5930cae7ad7269e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95c00c3d3a8f216d5930cae7ad7269e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95c00c3d3a8f216d5930cae7ad7269e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95c00c3d3a8f216d5930cae7ad7269e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_849f6198a566f8581e34999de3cbe595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8653a5fc28d4605b65c2035b2e29717(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58b88480a6a3a250a2eb80994a0049e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe1fe5417f9b51083fb847e9d229b827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c450ac0bffcd1ae4c39d873e273c0e39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc48eff68ec4077c8e1f2eed586f98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c6761d28fb6701fb0a16d9cf937890c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03970d371695ea096e5ec50f09a7338c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc48eff68ec4077c8e1f2eed586f98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_849f6198a566f8581e34999de3cbe595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fc0ca9e6ba14ea7124412e79343faa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b28ffc49829e64525707160793f1b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_14e7447ff50bec88a027dc4e6e570e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c6761d28fb6701fb0a16d9cf937890c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe1fe5417f9b51083fb847e9d229b827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ab49625ba9269bc081d59d08df1ff6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43dec9069fcf5af6cd70752fdeaa573b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_23fe17bb356b6d7375d0a418ed2bebc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_548e70a986ddfdd6e59284e665df51c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c4a10731c9c33703de56afbb658fe27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2864873320772382dc6d9415ebff5a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b983b1cc73b530e06ae8f5541f642b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5291f68250bd92253d89e06c571a0bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8a6312b07a3968832944def70532a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fc0ca9e6ba14ea7124412e79343faa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7f08611a442e51c4926e7ebb59c0fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a392b3b769f9db6c5f71bd23c8328ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_23fe17bb356b6d7375d0a418ed2bebc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03970d371695ea096e5ec50f09a7338c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5635388277fd20b53d2faf50afbe905(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_493d3f7e8f3386e812a75a9f79df63d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38453794e705fcacbd4f772ce2be8a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_849f6198a566f8581e34999de3cbe595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5deda1a8809f68c6d9fbe517805c336(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58b88480a6a3a250a2eb80994a0049e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e88a0ef74495a23644f2245868e68a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fc0ca9e6ba14ea7124412e79343faa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43dec9069fcf5af6cd70752fdeaa573b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3194da9927b515dac78d683d7ecc1b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef706ced8d290d632bbcdb653e5da9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_15903f68d285c1bd03837c2e9cac50b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9124d93e2163bc5e9bc29a98bf0779f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9124d93e2163bc5e9bc29a98bf0779f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9124d93e2163bc5e9bc29a98bf0779f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9124d93e2163bc5e9bc29a98bf0779f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b1ba137322a0c7f4f6ff2f9a384e8ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40394.7109375]], [[41053.18359375]], [[58203.19140625]], [[47845.3359375]], [[44826.7734375]], [[30734.732421875]], [[48187.421875]], [[27655.919921875]], [[62914.5234375]], [[37999.03515625]], [[43159.25390625]], [[48522.34375]], [[56747.484375]], [[47576.82421875]], [[71490.4375]], [[41356.09765625]], [[27665.529296875]], [[51633.05078125]], [[35305.46875]], [[64955.77734375]], [[49642.28125]], [[45967.33203125]], [[43274.34765625]], [[52577.59375]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2772ad017fbb3455bb6df7cab0f185fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32447.328125]], [[51521.078125]], [[55027.69921875]], [[56319.2109375]], [[51922.15625]], [[48465.328125]], [[44730.81640625]], [[44645.2265625]], [[36616.3671875]], [[60146.33984375]], [[75195.3828125]], [[49811.3515625]], [[69967.984375]], [[20346.373046875]], [[55820.28125]], [[50153.73046875]], [[55559.390625]], [[53131.55859375]], [[73189.2265625]], [[32012.06640625]], [[58937.90625]], [[50805.03125]], [[67007.3203125]], [[65413.1640625]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91b998e5695c3483e7c5ebccef4d2173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[70237.3203125]], [[51783.8203125]], [[41368.76171875]], [[36558.3828125]], [[74452.875]], [[41336.609375]], [[66183.75]], [[90987.4453125]], [[65005.82421875]], [[70354.6875]], [[80621.34375]], [[61925.55078125]], [[71502.203125]], [[58610.109375]], [[56892.04296875]], [[61840.93359375]], [[87014.0703125]], [[43695.94140625]], [[55268.40625]], [[51172.734375]], [[86102.890625]], [[47229.51953125]], [[55730.66796875]], [[67458.921875]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6fd60da99b3dfd64d3a7f80596b8977a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[68354.5234375]], [[57062.53125]], [[33374.52734375]], [[49705.4765625]], [[74635.0703125]], [[87549.1640625]], [[47589.55078125]], [[42842.4140625]], [[53218.18359375]], [[58559.18359375]], [[57287.1640625]], [[34865.0546875]], [[43199.37890625]], [[70856.046875]], [[54579.44140625]], [[39887.65625]], [[48813.6328125]], [[103218.6953125]], [[67022.3984375]], [[28652.015625]], [[82444.8125]], [[52055.05859375]], [[46388.23046875]], [[92584.9921875]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a392b3b769f9db6c5f71bd23c8328ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c450ac0bffcd1ae4c39d873e273c0e39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9124d93e2163bc5e9bc29a98bf0779f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_03970d371695ea096e5ec50f09a7338c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57e13dbfd70cc9444ddb86df8c763034
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46243eabb8d5b44811d30b94fc6cb2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_969da94d4c8e84574e268ca7c8a1f1fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()