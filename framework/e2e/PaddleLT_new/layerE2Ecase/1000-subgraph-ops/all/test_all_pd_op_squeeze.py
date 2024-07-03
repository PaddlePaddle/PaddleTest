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
class PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_107f56c6e4d588913f4a1fd34154fbc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61c115bdafceb29e1d9db9a2c1079820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d7542c39d37839ed6ff48e6279daa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59a76a3e32e1dd6d998b09d1c8a1eca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff529edc6a7ea3fe9579e2bc2f9408e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffaccecb367766210ca01da857167347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6aafc5a747379512c484bba80354f021(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_207980ec4ee3bde2c898e4ae7690b247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1af11ec433e8832fd4bc0beb861337a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f9752aaa33cb7405acfa059cb02633c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a9ddc3b8d8a3d155786782974c5dc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9508cbad018d159fa0713eb4455bbad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9508cbad018d159fa0713eb4455bbad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8d259641f2f72e31254975e65602542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f9752aaa33cb7405acfa059cb02633c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
        ]



class PrimitiveOp_b89a3e834947e5d74540672a672f4a52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_299b723e97868719942fe9db652357a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.34787654876709], [2.159102201461792], [1.9213602542877197], [2.2024269104003906], [2.1114273071289062], [2.1365914344787598], [1.959598422050476], [1.8337061405181885], [1.8550523519515991], [1.9595694541931152], [2.091510772705078], [2.2969653606414795], [1.895961880683899], [1.937199592590332], [2.1223039627075195], [2.2149817943573]], dtype='float32').reshape([16, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_85d7701d6d1551a715c9cf499bc2d653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.8714301586151123], [2.0343306064605713], [2.213071584701538], [2.2041175365448], [2.139944076538086], [2.1380159854888916], [1.9252699613571167], [2.2255287170410156], [1.8711376190185547], [2.096949577331543], [2.2395358085632324], [2.1725518703460693], [2.0819284915924072], [2.409768581390381], [2.231198310852051], [2.095301628112793]], dtype='float32').reshape([16, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efef2a412b068ef1c71bfa4eb0604e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d00b44b3ccf063c2a1831dc7b7a5b0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d91bd06b04f2a88bad189ce1946e2b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d91bd06b04f2a88bad189ce1946e2b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e7228768a28240c48dcdd698f2612fa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a9f8ad2079f824af9e7c814715aeabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc6df4cba07ff8dc2b66e338a0c11bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1af7d6c3217c4dee9c45d1f073f3d793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_996a9fbd445693fb99453b9b4df7aabd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_224239f4937ef82d7418d470195675f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_224239f4937ef82d7418d470195675f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd9492730ef8ae2ddd7405177c2a508f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b1e67b761e7717ef4aaaf584c81914a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a8aaada5dff752c6b8b57ebc3c897074(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f45b3f4fd9c62cfc265292d31dde6660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8aaada5dff752c6b8b57ebc3c897074
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_207980ec4ee3bde2c898e4ae7690b247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d0f03a988b1ae9571ee63d920a0f2b65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_173f641b083d61e1edc5571f5d40c236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_173f641b083d61e1edc5571f5d40c236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5994391d91b40b553a10c21572c7531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5994391d91b40b553a10c21572c7531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6d423954ef9ef462781c241f8e608dae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2137bc7d0371b20113034951b1d57ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffaccecb367766210ca01da857167347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9db240e678207c7a1d3e7446d6eb13dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9db240e678207c7a1d3e7446d6eb13dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11f01c6874025ec359a766989e8ad984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e915bda4de01e4b1c4ce6036b52d30c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aeb37568bf3b6046c996268b05d9e5e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8455597eb1c5b79d8c04dd2c54377a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f68c519f478da827a5623e4748afc595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f68c519f478da827a5623e4748afc595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1978b3a6b3d69dcf09159166c99e5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_591540425fd1bcaaa24cc68f10568b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_519ed4778a5c6c940f245dc18643ec38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b1e67b761e7717ef4aaaf584c81914a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40ad384b7c96da92a7b45ffa22c89e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e6b8c38f8c3cc4681ba20591d9f17b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1521999835968018], [2.1796727180480957], [2.1047537326812744], [2.159518241882324], [1.822599172592163], [1.9034342765808105], [2.1630988121032715], [1.9198029041290283], [2.1094632148742676], [2.0728089809417725], [1.9138891696929932], [1.9151036739349365], [1.9168546199798584], [1.9518407583236694], [1.93830144405365], [2.3321938514709473], [2.16713547706604], [2.033275842666626], [2.194241523742676], [2.034022092819214], [2.0413191318511963], [2.2838566303253174], [2.2300868034362793], [1.9731148481369019]], dtype='float32').reshape([24, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b39aca24fc573cf64c48945665cdfba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.222297430038452], [2.2627525329589844], [1.849575400352478], [2.053041696548462], [2.192687511444092], [2.159512519836426], [2.076664686203003], [1.9242665767669678], [2.056380271911621], [2.302868604660034], [2.204963207244873], [2.3179593086242676], [2.261704444885254], [1.8650039434432983], [2.3297932147979736], [1.9024587869644165], [2.0811641216278076], [1.9169301986694336], [2.161933422088623], [2.0917773246765137], [2.026906728744507], [2.07029128074646], [2.2546253204345703], [2.260570526123047]], dtype='float32').reshape([24, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa2901dfef3bc1fb52ad018ae5c0ab90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4a47bc9b4a2c6cc69b03e1f8ebd764e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6820c1cf70a593fcbe05fbd78958aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6820c1cf70a593fcbe05fbd78958aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5a62c1896bc14469121c3b4a2be5df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a08a20491adbebf6656d910ec231f763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_30dc45b8f9abc229ea35b02991c1ad5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1816418170928955], [1.8633242845535278], [2.1522037982940674], [2.2153337001800537]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e9087664010e9ee646844c33ea98943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.207533359527588], [1.9045121669769287], [2.0253593921661377], [2.0253467559814453]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ee870e33f925f598b57d475f69bb3ea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7e8f1da49d07aa940054186243abf17b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11b65baf5654885b138d109e789808aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e8f1da49d07aa940054186243abf17b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a76fc3bb1880dca9f04edda5935e4eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_792be3c32af524d948825149e84d1eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47366b3a76f4ad0abfd56156227ff637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_644d048fedb52b8dc54e371764e0c305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60d5e4c02866ec265b860fa66b7324d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b99135569539a39047579a6b4daca9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b99135569539a39047579a6b4daca9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_200875c625be6843aa3ed9aea59c5d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46ebfcfb0f4785441f2cee73b994bf68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4624f6702b19bcf454e672fa7389f390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1978b3a6b3d69dcf09159166c99e5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95295434e59fc04ffe76f1d21fe4e9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d84af74c6e832602dbd968de02bf010b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_751111bdcab58c5d8717c6aa1a34383b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a00925510295776f1511ec2d052a744f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a00925510295776f1511ec2d052a744f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_00260ea2287c4ec60ecc3e403d75be9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b126f4beb41cf32acea05c04106338b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b126f4beb41cf32acea05c04106338b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11cb274ef26705fa1c92fbf732ad4829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_df1a6fd9e3f49838c7a28a2d6f315c92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec7b1d0927c9697c50e424daf6586d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df1a6fd9e3f49838c7a28a2d6f315c92
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3a35f7cc8252f59575db121d4aa4e70f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9f80c945d81967929efaa7377d94b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a35f7cc8252f59575db121d4aa4e70f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9a944738cb1d453b26861a68d25045c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9a944738cb1d453b26861a68d25045c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1fc75feb0c2f6c311206ded2fec25855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_29965dae7c570a80fcb898d706dc6a73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8aaada5dff752c6b8b57ebc3c897074
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_acc23dbfb8a3e42e03fdec93d82d8d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3f7612df3520fb9d9ff1ce8d67df9b4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2ff7125298d7c0f9a708e33a21bf9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f7612df3520fb9d9ff1ce8d67df9b4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_52172819fffb2d5d2b985615540c6912(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe5fa9f9b4f87e5a52d35eac276ec45f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52172819fffb2d5d2b985615540c6912
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1885d8a6913e65714629db7b1917ca50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_afb2f1ca8cb23c55e6a0ef2ce1c471cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_127155040aa793f938a3855381597a81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4b018c8677cfa041d90c85fc3749afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0da873a450d8e29049b4e594309cf59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0da873a450d8e29049b4e594309cf59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4b46cd54e89b2504780bdf099bf5cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4b46cd54e89b2504780bdf099bf5cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ecfe1a068a12db215ef5705f52ee5b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ecfe1a068a12db215ef5705f52ee5b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ef11b37467de0ea27d6ad0d3a8b10f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ef11b37467de0ea27d6ad0d3a8b10f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b13b927d47c80be26e00d4643782d325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b13b927d47c80be26e00d4643782d325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceecb1588af84e1c1c24325513bd5b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ceecb1588af84e1c1c24325513bd5b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec3b791ecdedbe89ad2d9c2db98313ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cca7fcc9c2907f43cde7e75f37d7fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cca7fcc9c2907f43cde7e75f37d7fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8b588a249c7eb562751193488e9a3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_996a9fbd445693fb99453b9b4df7aabd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_021706673ac70ed43ce1acd0c5db14e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e56dc4d81bfc2472d74a12d51709873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2a891e3649a4202cb0e4b361a3e4c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_190879faefced1b4aead33e65c7d661c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c373055c43fefde454415cd829f10df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58c7101c476a0c7c1a7f5ba49be8eceb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1815907955169678], [2.2769126892089844], [1.907932996749878], [2.0796091556549072], [2.110363006591797], [2.1431312561035156], [1.9416979551315308], [2.245042562484741], [2.209714651107788], [1.8555011749267578], [1.881988525390625], [2.0004849433898926], [2.0412535667419434], [1.860338807106018], [2.17122745513916], [1.93278968334198], [2.2222325801849365], [2.042781114578247], [2.1074957847595215], [2.1499881744384766]], dtype='float32').reshape([20, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6fb89d316fdcb015a0285674f028a5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0350992679595947], [2.1021230220794678], [2.1693103313446045], [2.033270835876465], [1.9085214138031006], [1.9629747867584229], [2.2445266246795654], [1.951501488685608], [1.934584379196167], [2.248936414718628], [2.3342783451080322], [1.8598324060440063], [2.115058183670044], [2.0682835578918457], [1.9403040409088135], [2.1461122035980225], [2.099975109100342], [1.9092484712600708], [2.2656540870666504], [2.1662161350250244]], dtype='float32').reshape([20, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95295434e59fc04ffe76f1d21fe4e9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46ebfcfb0f4785441f2cee73b994bf68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f9752aaa33cb7405acfa059cb02633c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9ede36e37cf011fa1c359a0c18abae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df1a6fd9e3f49838c7a28a2d6f315c92
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe65f5022636e7ff32a69a8aefa7bd31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a35f7cc8252f59575db121d4aa4e70f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_751111bdcab58c5d8717c6aa1a34383b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a9aab990d4cac2047e772b11a382e39e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a9aab990d4cac2047e772b11a382e39e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8c3e55cf024c69778b356811cb2217c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_200875c625be6843aa3ed9aea59c5d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e24d3dfd6c6d4ab57f2dfde11cb32ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e4e3978e071e98d9594baa40230ed77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a84dd087503e62431c584ca7845a0b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a84dd087503e62431c584ca7845a0b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_419410e432cb968e0ca2f2d4b6d640ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6abf0f7a52967587bf158fab003e900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f7612df3520fb9d9ff1ce8d67df9b4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_200b8ee4767cf6cf1965aece8b01db51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52172819fffb2d5d2b985615540c6912
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_463980bce97715e9b1efcb56942ab82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3fdec699ae8620f465f251a1d9ad5f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5335da310ab758484d6cb970338ee225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02e14fd0ee3c257a7a5f1118372cef95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae8e12185159bfd8ce466a1f9130e761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f6d879e8e8806d8c9dd748eea3c8f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_cb41464438e84d3938aae4495fd4e2af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56f93cf119ae4079622ac343e942ddc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ee17f0e727de660ef580b350eccdc67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb94ebc53d62fa11f4a3a51a16ac1e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fd000a881d3eebe96b617257191336c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7d9f09f9d801fea46432cbb1eec82e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7d9f09f9d801fea46432cbb1eec82e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_70adad6afa5c6c4a78d20057bb91ea86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb94ebc53d62fa11f4a3a51a16ac1e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
        ]



class PrimitiveOp_c8abc660aa844735df7f79f4152a427d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c506cf5a1ec3f15ae550c0de8544b78d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.34787654876709], [2.159102201461792], [1.9213602542877197], [2.2024269104003906], [2.1114273071289062], [2.1365914344787598], [1.959598422050476], [1.8337061405181885], [1.8550523519515991], [1.9595694541931152], [2.091510772705078], [2.2969653606414795], [1.895961880683899], [1.937199592590332], [2.1223039627075195], [2.2149817943573]], dtype='float32').reshape([16, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2aa9b63bafab3878ae0d3e3192b5ebac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.8714301586151123], [2.0343306064605713], [2.213071584701538], [2.2041175365448], [2.139944076538086], [2.1380159854888916], [1.9252699613571167], [2.2255287170410156], [1.8711376190185547], [2.096949577331543], [2.2395358085632324], [2.1725518703460693], [2.0819284915924072], [2.409768581390381], [2.231198310852051], [2.095301628112793]], dtype='float32').reshape([16, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca7675f56b1e80850c670e8f5e396c97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f3fef195c76a7041da7ad8baacbbecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_883b4062ceb73368385ec3ca83d65e32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8732d397cd6083a774d13656a4e385ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8732d397cd6083a774d13656a4e385ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2389daa67bfd6ba953cd3a5d488a65b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a98c4471ab28e2277fb4e137b92eecf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4f71b328b3726b0dc46b0aff14fdf0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c09ba1238a9f0651d15e1198e76696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffe935da27b8424a7a426dfbca78efce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffe935da27b8424a7a426dfbca78efce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f1983aaf0f2f2cc45f4d55aa2021997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c0409136730c7d58ef06e176490e43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b597523a04a1961c3a48e94336450c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_56f93cf119ae4079622ac343e942ddc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_511ebfde8c6216a4a237a1836c69ac8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bd2b3f3a7cd0dae0046cec85fa03d29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bd2b3f3a7cd0dae0046cec85fa03d29f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3bdf8e51cc45fb0769d08d39d55ac351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3bdf8e51cc45fb0769d08d39d55ac351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ee3fe9864a798604907bf8f2ee66a796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f6d879e8e8806d8c9dd748eea3c8f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b394fc6ca83f795ba5c89c947e780fc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b394fc6ca83f795ba5c89c947e780fc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4953b8c6f122350a8debd015d1bfd97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6cdd0fc4711249b0b38d895ae18dfaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddeab4cd6a9d3c90860ae0533ba1ccbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5fb4ae00e96e60b7e57aa2f857525aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65cbee7c274f5376c827d9d1d3568f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65cbee7c274f5376c827d9d1d3568f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_954bfac9bc21a333f24b06008f6dd186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5facc11e8b6ed63bff5abded60dfd679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93a01b6c0a984b7df39dd2f4729e5b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c0409136730c7d58ef06e176490e43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0f874a29a642d876e8156db7d4f3fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2685d600ee03e050d1fe72e91ad8fc72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1521999835968018], [2.1796727180480957], [2.1047537326812744], [2.159518241882324], [1.822599172592163], [1.9034342765808105], [2.1630988121032715], [1.9198029041290283], [2.1094632148742676], [2.0728089809417725], [1.9138891696929932], [1.9151036739349365], [1.9168546199798584], [1.9518407583236694], [1.93830144405365], [2.3321938514709473], [2.16713547706604], [2.033275842666626], [2.194241523742676], [2.034022092819214], [2.0413191318511963], [2.2838566303253174], [2.2300868034362793], [1.9731148481369019]], dtype='float32').reshape([24, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cc151d0bdd248c9a5d6d01a0a436f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.222297430038452], [2.2627525329589844], [1.849575400352478], [2.053041696548462], [2.192687511444092], [2.159512519836426], [2.076664686203003], [1.9242665767669678], [2.056380271911621], [2.302868604660034], [2.204963207244873], [2.3179593086242676], [2.261704444885254], [1.8650039434432983], [2.3297932147979736], [1.9024587869644165], [2.0811641216278076], [1.9169301986694336], [2.161933422088623], [2.0917773246765137], [2.026906728744507], [2.07029128074646], [2.2546253204345703], [2.260570526123047]], dtype='float32').reshape([24, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0296e20b4c66637eba3a9dd27bbd623d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90d9611a6eb0dff9d3eb213614abf901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ea8232472581db9185aa81abc53dc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ea8232472581db9185aa81abc53dc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d520e3988bcd6ce25132d8f0d7a3c44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f1610b4fa26915daf6a11b5c26e1ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44cd74138065b8daec96dd60167c551c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1816418170928955], [1.8633242845535278], [2.1522037982940674], [2.2153337001800537]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ae43f034f1888193fe96b16a4bcc157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.207533359527588], [1.9045121669769287], [2.0253593921661377], [2.0253467559814453]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_33710390fee8418e9de45b75cc929078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc02d45b36a1ddbaa8c3fc23bdb72578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dbc2d4453a8f0253f1c78d7e6a51d849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8bb231543ea167c59e4636179b06670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b23cacef947e56f7cc143472ab834b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fa3e4543a0273d5303cde003350cb39e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76d58ec9686d2b07fe302903a22e3c4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e998c0a1a3b4d146cfd27c1aab28368f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e998c0a1a3b4d146cfd27c1aab28368f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d8081fcd6c0f9e5e7792d394a329f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3dd28cd2fac5aac9bff1565954279c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c418de12de37eea5d8dbd1394791760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_954bfac9bc21a333f24b06008f6dd186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_491cadfe7acb454e7cd9a7bb65c08f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5ddbbf51553b7ad1500b465570d8938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fa455f72ac56584f626f17ec2f0b77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89edbedb1a09ecb71a1de2cce504f00d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89edbedb1a09ecb71a1de2cce504f00d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78b1efc28aebfe35fdc80ec2d3e37e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b75c7ee3c4ebed2ef7fd45eea6f3393a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b75c7ee3c4ebed2ef7fd45eea6f3393a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11cb274ef26705fa1c92fbf732ad4829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b6c0bbceb138ef8a128e5c458a5bee1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8428a6729f99ba67bba319de58dc7901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7de54a09fbb6b6389d89d325eb369ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8428a6729f99ba67bba319de58dc7901
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0403f0e40acf84af19f90f160a6e9713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0403f0e40acf84af19f90f160a6e9713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5220228431db5661fdcf9489e23f6748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02cde49b67196e2951a59991b2262a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_798b33e970b7441fefc93a2a13e897c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9c3f13317024517a9068de8594946faa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_213669359003cc149d24366a85686806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c3f13317024517a9068de8594946faa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e72659ad328a84cbf860c60be4515145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f4057b67fc4dffbb5025c54a474ded6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf6de3c5eac246360880f3dcbc1074d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92073823f2b706fec35eb23e4ed50d46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4422b810c025519c8ff9d20276ddcd9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a6443a21023a1c981b5b35297f8b6a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a6443a21023a1c981b5b35297f8b6a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e412084b6f7015ad738fda2a8883c198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e412084b6f7015ad738fda2a8883c198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0d0f07fcad0f680e19fba854c526c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f0d0f07fcad0f680e19fba854c526c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59b8ea226871297b80558b5fc58fc02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59b8ea226871297b80558b5fc58fc02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b108d52d250110613b63d8b0a480c1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b108d52d250110613b63d8b0a480c1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3396588d309b55289228e571f89558e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3396588d309b55289228e571f89558e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6624b86c7fd1dc07b0e47aa01bff95ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29eabf4987eff07da81a477052869e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29eabf4987eff07da81a477052869e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13abaf06f916901d1b041dd5a4458aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47c09ba1238a9f0651d15e1198e76696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe67c8f5a6d68f16068826e9656d92fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d92769d2f1bf3b42588b87e1b85af57d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2a891e3649a4202cb0e4b361a3e4c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367324e02664a301ad4d638057893f0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_879cc6cdf15daa6f97276f93db0fec57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7711bd105379b2974a395c184e84d008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1815907955169678], [2.2769126892089844], [1.907932996749878], [2.0796091556549072], [2.110363006591797], [2.1431312561035156], [1.9416979551315308], [2.245042562484741], [2.209714651107788], [1.8555011749267578], [1.881988525390625], [2.0004849433898926], [2.0412535667419434], [1.860338807106018], [2.17122745513916], [1.93278968334198], [2.2222325801849365], [2.042781114578247], [2.1074957847595215], [2.1499881744384766]], dtype='float32').reshape([20, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11357a49892be0f842a62711a823a7a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0350992679595947], [2.1021230220794678], [2.1693103313446045], [2.033270835876465], [1.9085214138031006], [1.9629747867584229], [2.2445266246795654], [1.951501488685608], [1.934584379196167], [2.248936414718628], [2.3342783451080322], [1.8598324060440063], [2.115058183670044], [2.0682835578918457], [1.9403040409088135], [2.1461122035980225], [2.099975109100342], [1.9092484712600708], [2.2656540870666504], [2.1662161350250244]], dtype='float32').reshape([20, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_491cadfe7acb454e7cd9a7bb65c08f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3dd28cd2fac5aac9bff1565954279c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb94ebc53d62fa11f4a3a51a16ac1e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6cd5d2d104edc0075ce6e7265b629156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_70ebd1d32ee7cbe80ad751bf1c4675cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8428a6729f99ba67bba319de58dc7901
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4fa455f72ac56584f626f17ec2f0b77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7080680591abf747491b71a006527e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7080680591abf747491b71a006527e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_123c07f1dad265be03150f516702f87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d8081fcd6c0f9e5e7792d394a329f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc870390d6629627571b4503779d2ff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd48a84f6960beb8745e7c4fde126ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e691649068aee6714b589e599d2cd979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e691649068aee6714b589e599d2cd979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_32daabdcaa62c0c42e09a67246cc1862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da5f2d715bd1b362beb542b8e8ee79d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c3f13317024517a9068de8594946faa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f58f82994e0bb654ec0dee4a888a3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()