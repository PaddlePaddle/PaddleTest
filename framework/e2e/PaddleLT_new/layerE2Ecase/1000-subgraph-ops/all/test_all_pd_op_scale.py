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
class PrimitiveOp_088fa044ab700af81b3be135f754d3ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3458a6070a1c3bc46419c696b851044a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_088fa044ab700af81b3be135f754d3ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_294384b8c13ec501ee28a6831118054a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67a97e96f64f74af411b7bef53f781d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_294384b8c13ec501ee28a6831118054a
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_33867d0f525c39aecc73ac773144aec3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a35df2cb44a2c2f2ca9ba152a11f6a77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33867d0f525c39aecc73ac773144aec3
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8a1f91064da9c56164e1870ca2c682bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.85, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae3684875ea5cefb955a4729ca9eede5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a1f91064da9c56164e1870ca2c682bb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2914212942123413]]], [[[0.39375486969947815]]], [[[0.16414831578731537]]], [[[0.9359673261642456]]], [[[0.29178929328918457]]], [[[0.3484293222427368]]], [[[0.662407636642456]]], [[[0.027838630601763725]]], [[[0.07953529804944992]]], [[[0.027762018144130707]]], [[[0.6288183331489563]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]



class PrimitiveOp_4996152f05746018494ad632380a445a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.1764700412750244
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f44c7c5133a47ebd203a168c4cdd876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4996152f05746018494ad632380a445a
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_faab6d0500e164d1782c67c86e887c3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3c2dc9cedbef2b0ab25c21080a4a5e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faab6d0500e164d1782c67c86e887c3c
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_947e9bf7f3561d180d896d73968e26fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.1428600549697876
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_658fc5a2490bcc91dadc884755c56347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_947e9bf7f3561d180d896d73968e26fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.5
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72eced3457b18cc1b955150fd7ff392c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2106356620788574]], [[1.2134649753570557]], [[1.5126121044158936]], [[1.2061398029327393]], [[1.3825193643569946]], [[1.2376539707183838]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd74bc5b188372ed2a29a5d91fb5df65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.393440842628479]], [[1.0911762714385986]], [[1.548488974571228]], [[1.0763018131256104]], [[1.0530097484588623]], [[1.4913479089736938]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92744003fcfc504433a5f6c55c01db79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_61df107e237a575a6a230d9e977d1186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_749f0500f667bdf6c86747987f6fc03d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e31d58a844c105b55f981ae123bf934c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4669b1df1a883cde41737b5a4b48869c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c3ef78629590f6107c4e398da8d0f493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_cd66f65bc5bc41b742b4ea78073776ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.10000000149011612
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17c1a17fe4b09bfaa80ef1fa9cadd159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd66f65bc5bc41b742b4ea78073776ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3268720507621765]], [[0.1784019023180008]], [[0.18456639349460602]], [[0.1489272266626358]], [[0.420693039894104]], [[0.3033420443534851]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8969f1ab87d91fbac8ee0f1d7a4104d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd66f65bc5bc41b742b4ea78073776ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.08347339183092117]], [[0.13196611404418945]], [[0.09201230108737946]], [[0.3147572875022888]], [[0.2593747079372406]], [[0.21239152550697327]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_ce78b0c22c5c73ec83998383f15f05c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.20000000298023224
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_056181211f61010f619933f116630d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce78b0c22c5c73ec83998383f15f05c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0281860139220953]], [[0.44767507910728455]], [[0.08558890223503113]], [[0.2602667808532715]], [[0.10315723717212677]], [[0.27075910568237305]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc5a0ed1a29b00c52198e31eacefee81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce78b0c22c5c73ec83998383f15f05c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.45546549558639526]], [[0.3317403793334961]], [[0.34772875905036926]], [[0.4843132495880127]], [[0.25377902388572693]], [[0.41150492429733276]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.5
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b38245720d0d41244176aee139ab2674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 9.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b337c9a5250c040cae55bb1d1ff37d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5134b984a39062791624c7d43726b62d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -0.0555556, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc2ad2a3179e747859ceec88863864d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f946b3a9fd9f7f31b5a44b7e48771f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_040f39aaddf887ddff3436a738e833a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab7e67481c1d26daf83ff4b6d5879155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
        ]



class PrimitiveOp_7f74b2c6640d727f68081e7375b735ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2d4a7b2f6c3b9d11dfb27709c8fc7a3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f74b2c6640d727f68081e7375b735ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2437746524810791]]], dtype='float32').reshape([1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_214a87bc9823ae0e4f99f421c8212707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_819fa67ec5dc8d066f1895000f2d5875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d0a6ef13c05a14b12c2695fa68cdccea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_aa1435d87bdcb643ee9d40934d3f3a92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_733da9760cdcdc29fa7550850ceb3b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa1435d87bdcb643ee9d40934d3f3a92
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_07efebac16ffc35022340f0d4b49b9b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1541451f9f59d440cb74a8858e2a592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07efebac16ffc35022340f0d4b49b9b1
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8865b3592a7988dee829e08748725a43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eaab3ae492d6c18debed80ed2c17677c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8865b3592a7988dee829e08748725a43
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_106d4f6884ea9079d478ea9a7f9aea2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43932d2b975a6927fa8286eca087365c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_106d4f6884ea9079d478ea9a7f9aea2c
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2164ec58012afe10267bc6829821c80b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1935e06b13c303527525b3ad056c9939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2164ec58012afe10267bc6829821c80b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]



class PrimitiveOp_465d7d6b31aebc7098f4bfe0675860d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61b127a6acc49a7536942965a832a4bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_465d7d6b31aebc7098f4bfe0675860d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b63c694f540ca74d88d798d7a50165b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6071fc799ab703757b431449660c22eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ecbb007ebffe608ceac85e5a6ad3cf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6071fc799ab703757b431449660c22eb
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1541451f9f59d440cb74a8858e2a592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07efebac16ffc35022340f0d4b49b9b1
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bbf25d65df564899648c759dee1c94ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_425715d91d09c03f6de55c1366a4843c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf25d65df564899648c759dee1c94ef
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_43932d2b975a6927fa8286eca087365c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_106d4f6884ea9079d478ea9a7f9aea2c
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e10d59034017db7df19c153856859ee0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f26b7f17bc41e128cc8e583e592d73c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e10d59034017db7df19c153856859ee0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1882213c65a79728f850b761ab65f65c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_465d7d6b31aebc7098f4bfe0675860d3
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8b561adc88fd6f915d5e5acc529efff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2eb41fa673c7a5c9e17465e8c3fd002c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2.5
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e7e87107cad0b026d58a20985570358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4fe9ecfa90d725c55a7b0728a3e8a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66c675e75b3239325aa3771ad588f437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4b6d970b8e038568d92d479283daa5d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64bffea55a37388ce26c0bcd427efad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(1086.693603515625, dtype='float32').reshape([]),
        ]



class PrimitiveOp_b674146c71585e699d219730cca623b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.00390625
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd9c2163fb338ecfcb122a4fa0641997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(180.47015380859375, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44abe3fe6d4864f8e914e4d46a6e95b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(5.625393867492676, dtype='float32').reshape([]),
        ]



class PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1e-09, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d63880fa44727551d52cc013b81c78ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.04387051612138748], [0.006444363389164209], [0.0013840901665389538], [0.002383660990744829], [0.010043813847005367], [0.012843223288655281]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9176c689fe1a42673312857add8ad927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0013636639341711998], [0.00024246015527751297], [1.1780663271565572e-06], [0.0015065169427543879], [0.02632260136306286], [0.0008185577462427318]]], dtype='float32').reshape([1, 6, 1]),
        ]



class PrimitiveOp_a6b5b27ef4cd57092c72a992645f6099(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -6.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7cb9a92581d925c2df0d8e1c99cdc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6b5b27ef4cd57092c72a992645f6099
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a3b86a311f3bdd78377711f49c54c5d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.08333329856395721
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f4a688ffd499534dd5515c405f705e8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3b86a311f3bdd78377711f49c54c5d7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4026823937892914], [0.07231664657592773], [0.011129551567137241], [0.07582680135965347], [0.40325236320495605], [0.13427451252937317]]], dtype='float32').reshape([1, 6, 1]),
        ]



class PrimitiveOp_c14f6b5e31adc9d8a528554c7030bfb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 6.28318977355957
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7324eca285978a5c09248ca9dc91670a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c14f6b5e31adc9d8a528554c7030bfb1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.033556852489709854], [0.006026384886354208], [0.0009274622425436974], [0.006318897474557161], [0.033604349941015244], [0.011189538054168224]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b15a5a4a7dcbd90001a48ca4522161ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f74b2c6640d727f68081e7375b735ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.21084406971931458], [0.03786491975188255], [0.005827421322464943], [0.03970283269882202], [0.21114251017570496], [0.07030598819255829]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa3109363ce374c82889b389a3686dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_672b1db58f1b08587df4e69f739a36cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6675bd164c9df9618cddca418e6d69c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8873c4375a69d4a03a9e1aaa3ff27480(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5ef2294784f273bd8f300032c02470d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8873c4375a69d4a03a9e1aaa3ff27480
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3065ca0207c6c11804a522f0c71ea94a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0958900451660156
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59e45c172d8d69bbee207ad02990be93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3065ca0207c6c11804a522f0c71ea94a
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b782107122d84e68620d7bb3b3f7490f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_294384b8c13ec501ee28a6831118054a
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea873f1632c418d73a1421ff626c3b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
        ]



class PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.25
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9f670c5b148606fccae5c3428d23ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1677451133728027, 2.1457746028900146, 1.983431100845337, 2.2028181552886963, 2.1208109855651855, 2.1369168758392334, 1.9550294876098633, 1.8417924642562866, 1.855881929397583, 2.023456573486328, 2.1109743118286133, 2.241074562072754, 1.9125232696533203, 1.9445034265518188, 2.164375066757202, 2.1581637859344482], dtype='float32').reshape([16]),
        ]



class PrimitiveOp_739dc0984d342d6c707069659dfcc435(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.25
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05a03ce5a86b3b5b4b76a6fd7e377a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(1.9255192279815674, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1582c729164d300befc0f03f18256d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eecc743746b943c12d4cf3f61e97474b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_393732e3ca1ca09a9fd943c633fa5e51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d72641a11818727974ccbecd0afe065b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.5
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_53a85999766846a506c56fadd9f08b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d72641a11818727974ccbecd0afe065b
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_53a85999766846a506c56fadd9f08b51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d72641a11818727974ccbecd0afe065b
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18ad356d9b7221c5d3e2b3c28b47d985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92744003fcfc504433a5f6c55c01db79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_395ca1124158bb0c79682d510448e780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_263d9b3686b6d2e6cd254ca409f29bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39fca2bebd4243f221f8a3f0503b878d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_148fcaa9483982d88458c05fff486637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68dd3720e439087864c85e52e5305bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(33.84434509277344, dtype='float32').reshape([]),
        ]



class PrimitiveOp_d44e8a38873aea2db72931fe6f7c4daa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_526d09adc67a94a36b5e6338aaac3b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44e8a38873aea2db72931fe6f7c4daa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8d10799fbc399eee45dc775e493f1be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7785f31dfba0bf596e4adf759b14dfe1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 80.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2abe6d886794017b4873f0c2cb24b818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7785f31dfba0bf596e4adf759b14dfe1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
        ]



class PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_97cb3602b7321ae480e907496b902331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c11ff0c29dd14d76cce66a10bb05cf29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3d249d43fe80355bed1630c616d284b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2bc635de2b2b6d279db476288decf4b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e13a861f5707de200b1131006cda5013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f437d53fcba7e67ddd9be52c5748d463(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7de054f24035afaa604ef77cd091f270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f437d53fcba7e67ddd9be52c5748d463
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76e08aeb734119e433f06895a9b697b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76e08aeb734119e433f06895a9b697b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e37e3df9ee18a6b71f9401b5bfdcfc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc17b456c8368a7d5d097ae558f8b939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d2d288c654765f4b64d37732648cc700(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9081993de8731ecd108a14a3cee4e27c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]



class PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b6115ea18b63f9fe2e34960b0dff3e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735d90b0e8c7ba4be87cd5734fa573fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_735d90b0e8c7ba4be87cd5734fa573fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d7e2742e1387d50bc454aa464469833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1902f8bba4888d920a06185974cc514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_786a9171be33e82507d723cd50c57126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faab6d0500e164d1782c67c86e887c3c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8137500286102295]]], [[[0.23146536946296692]]], [[[0.6546924710273743]]], [[[0.5924586653709412]]], [[[0.720824658870697]]], [[[0.24277272820472717]]], [[[0.4453102946281433]]], [[[0.6353831887245178]]], [[[0.4441324770450592]]], [[[0.6455785632133484]]], [[[0.2848832905292511]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac044c2fd080bdbeb2615e855eb8982d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_947e9bf7f3561d180d896d73968e26fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ffaa6fdaa22f75a41a46021c19df9c57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eed0b2aefbd75e040d5e877792a22954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffaa6fdaa22f75a41a46021c19df9c57
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.001793922041542828]]], [[[0.24888120591640472]]], [[[0.15599043667316437]]], [[[0.9758208394050598]]], [[[0.9929758310317993]]], [[[0.08918146789073944]]], [[[0.9301931858062744]]], [[[0.4761162996292114]]], [[[0.3519049882888794]]], [[[0.513953447341919]]], [[[0.2637135684490204]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]



class PrimitiveOp_1bbc8818a02fa79df132b41b02f5557f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0526299476623535
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fac9c66ac37139dd16015218df8ae4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bbc8818a02fa79df132b41b02f5557f
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6aa7b7d93b00ca94675f8db2b6b1bd2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 9.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe18f846286e29dc9291fa4760773b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aa7b7d93b00ca94675f8db2b6b1bd2a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd3cf90a1a9db370da0a320ff2f252cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0b2f40668ded1f14ad11532b48914c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(110.48657989501953, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a651ab8f784078f2a60c7482d4faec9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(3.6162848472595215, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddd5e30a73a679891d48f2c1e5ccee00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7cc2db59c68fbc3617f87b7b39e0c5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd012b37a24e162e8808b0c06f7efff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a1bc86bc5f58f28a870cebef32b0256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_30f2f6cd9872451c57909ed349da4936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_32e8d5154c79d485e45c3fcd293beb7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_527415823150693123431d4b4c5bb100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32e8d5154c79d485e45c3fcd293beb7c
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3f8e55dbba9cc11ba619ac30f4dc4a43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9f148e7f8b4a05d833c30403645dad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8e55dbba9cc11ba619ac30f4dc4a43
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ffab3402a6a57f664d7b984050fbf8d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_86dcd86efdf101d2d9b5ba89e324b077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffab3402a6a57f664d7b984050fbf8d0
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_96125b254f4174dca4e1d031c6101f5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3a0a9495df20df4d33967093a16669f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96125b254f4174dca4e1d031c6101f5a
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9b3aca943b354d98ff99a8ea73af96bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e49bfb6026eca3fbc0e0b3addcbd7b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b3aca943b354d98ff99a8ea73af96bc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]



class PrimitiveOp_3656b8c2a1548b6fbf7f943c2c8212a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f4de62e70107388e01c9c69e5255065c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3656b8c2a1548b6fbf7f943c2c8212a0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77747db0509fc2944c1be95b1d1dc40c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_85cd290edd11ed4c29379ada3bfdebf3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf906086e2bfb091fbb7cb61751797d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85cd290edd11ed4c29379ada3bfdebf3
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9f148e7f8b4a05d833c30403645dad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f8e55dbba9cc11ba619ac30f4dc4a43
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_291d47f21a7bfa0642b4fdc537ec4f9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cb2625e047011eb92afa48aeafb34be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291d47f21a7bfa0642b4fdc537ec4f9f
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3a0a9495df20df4d33967093a16669f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96125b254f4174dca4e1d031c6101f5a
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f4d39ee4bdbd3a6b4dc8e965b56c7882(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_25f352ac12c758e32f57f1d9d6a1985d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4d39ee4bdbd3a6b4dc8e965b56c7882
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c347206ba7d4a8c28b95af196d753ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3656b8c2a1548b6fbf7f943c2c8212a0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17c57f0ec6c46d2af1e4054b4cae0a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7785f31dfba0bf596e4adf759b14dfe1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3fa83374d765ac4c4457bfe02d9fdb1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76ed9045f2b3f75632f967c05d2a3cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(182.99789428710938, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_35fd57c66ada64b4069ec7451cbe2165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(3.0948688983917236, dtype='float32').reshape([]),
        ]



class PrimitiveOp_d21f94561a0c18c769ffc109fc0a4336(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a4c2d428fd62ada4d8ccd50eadf0eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d21f94561a0c18c769ffc109fc0a4336
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_41cc2c63c8cec85a926ab2a4e2ebbd8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0256400108337402
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f6cf0d15be24daa0af110896dd22dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41cc2c63c8cec85a926ab2a4e2ebbd8c
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4669b1df1a883cde41737b5a4b48869c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9604c8ff95876414b086347ce35e0e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81b68d432d25287aaa42129b90be1605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ebc127f2cc30fe443dff87b2144036a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d0b2784c43ecdb9b46647c379e3aabe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_149306fa7f6a30a6de00702b100b8b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_612c23b2bdbfd224d9b0af3a8fc80846(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.01010000705719
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3495fb43ea9d40660329e055376762a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_612c23b2bdbfd224d9b0af3a8fc80846
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b835ddec87a75bc248bfd9206ade56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.023871352896094322], [0.0072302743792533875], [0.028566552326083183], [-0.012393927201628685], [-0.014890296384692192], [-0.004542335867881775], [-0.037730999290943146], [0.046309053897857666], [0.10882711410522461]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5cad223a778bf47060e386961af8e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0006073885597288609], [-0.018595393747091293], [0.022392649203538895], [0.00693031121045351], [0.07298476994037628], [-0.014844009652733803], [0.013983488082885742], [0.0004492272564675659], [0.022267237305641174]], dtype='float32').reshape([9, 1]),
        ]



class PrimitiveOp_235493aee458c3fb166ca477f768df3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8c68d651cc6360837e76864edda2792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-40.3016242980957], [-1.388820767402649], [0.27571114897727966], [-2.788365364074707], [-1.20401930809021], [-0.6939953565597534], [-3.698253870010376], [102.08600616455078], [3.887320041656494]], dtype='float32').reshape([9, 1]),
        ]



class PrimitiveOp_87736f1f5873a74b917244bf139b916c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c85dd77d694b1d5e0857acc172bc356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[41.3016242980957], [2.3888206481933594], [0.724288821220398], [3.788365364074707], [2.20401930809021], [1.6939953565597534], [4.698253631591797], [-101.08600616455078], [-2.887320041656494]], dtype='float32').reshape([9, 1]),
        ]



class PrimitiveOp_664f9b3b573a9be9c841819852cf4800(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c719907fb5d7a9c391f15bb9046ec9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_664f9b3b573a9be9c841819852cf4800
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3393935f880a5162958eb62b07cc2842(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.09090910106897354
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_abb85589745943fc4cdee303d090237f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3393935f880a5162958eb62b07cc2842
    def get_inputs(self):
        return [
            paddle.to_tensor(11673.541015625, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f8eed46f1f6eec86205985dcaf04588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(1061.2310791015625, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c90e773c456927142568b0d6d1c77f91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.029102390632033348], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c16ff9b6503bf9abad6539aee42fc7fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.4080766439437866]], [[1.0083612203598022]], [[1.049621343612671]], [[1.0553209781646729]], [[1.6198089122772217]], [[1.0849533081054688]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9541c0d976646ae499f64bf41705104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.436981201171875]], [[1.051492691040039]], [[1.3073549270629883]], [[1.3907482624053955]], [[1.4530985355377197]], [[1.1249167919158936]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_b248181357313e26cc5891d0e1356171(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 128.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e3e539de5658b23dbf504d651a8d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b248181357313e26cc5891d0e1356171
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_58b7eb7379f4736e8935d32d7d51634d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d85465a5c3a76622b0502e562c0d0a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58b7eb7379f4736e8935d32d7d51634d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78e3e539de5658b23dbf504d651a8d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b248181357313e26cc5891d0e1356171
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4d1e50972629d13989c6c6fb8bf8293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33867d0f525c39aecc73ac773144aec3
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9330db3991ce01bbb9d3e3276ad678df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9330db3991ce01bbb9d3e3276ad678df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e4a292f9296dca7d71a97a9a1bdfba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_219c960a660fb8043e3377f7cacde5da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ddab0b3ae3068aba12d1055768064da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3494eccd83a35d83779e21e74d3a145c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a45e52033e636bb47baff5efed42e1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a45e52033e636bb47baff5efed42e1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]



class PrimitiveOp_698c9603cf85282204f973df84f4df7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.111109972000122
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a072b0448d8bb76bc15cb65daeec27b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_da4529d95e6b70f50c927674c12c9bb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.5
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0025bbfe895a090a34cbca3d72fea373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da4529d95e6b70f50c927674c12c9bb7
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a20f5f11713f1292e5918afe693ec99f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da4529d95e6b70f50c927674c12c9bb7
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ad6376dd5b56333ff206f5fd1e34b13e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_adcd82a434b05165f9d20d1c3be3751c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad6376dd5b56333ff206f5fd1e34b13e
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3f29ce3110ea8d2ef79166ca74b3fe7b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.1940300464630127
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4d23bd3db3fb3aac9596993cc6471c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f29ce3110ea8d2ef79166ca74b3fe7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c45d0e57de87145d0dadd9fe5052049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(89.05722045898438, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9ff87c32fab6c6f85580759417cb4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.33624267578125, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aaafcef19a8869b19e67f0510ad68a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4596b46ba897294bf283bdd5ff9a513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ee6be1e392e9a3e5d62b265f7c7ad13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(6.2698893547058105, dtype='float32').reshape([]),
        ]



class PrimitiveOp_b13cc3a42c6bfa985e778e5785224fb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.111109972000122
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddbd0ce523ec46b97e4977dc42fbb09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b13cc3a42c6bfa985e778e5785224fb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb99abd90a079e529b970f32cd7e0e57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3087916374206543], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a9102543375b5d29f8c334365ec381d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2936667799949646], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.5
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5c0533446fb39c546e09df57e5710fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8984315395355225], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec2066eda573d0af5a97bad3b2ece9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f83777f27331e8106baf3bf44b1f6a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da632f6c8b7e41aec56de7cd63eac16c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_604d3e059d7d3b6ee2e24da9fd2ff72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(108.77389526367188, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4cf957b44d7fa15b9ead63d26adac84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(62.76165008544922, dtype='float32').reshape([]),
        ]



class PrimitiveOp_17e45e927e55af3d09c8c4aa64947bfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d6ab184c6251ad74e5f7a313a8b894da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e45e927e55af3d09c8c4aa64947bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3495fb43ea9d40660329e055376762a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_612c23b2bdbfd224d9b0af3a8fc80846
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_908a6a3d3c1505d8bc8b2fce3c348c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3636210858821869, 0.860653281211853, 0.45989561080932617, 0.293732613325119, 0.43766167759895325, 0.5128899812698364], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_766d0fa74fc5f089bcb5154172af5880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7485138177871704, 0.23536936938762665, 0.8053454160690308, 0.817694365978241, 0.6373422145843506, 0.4877123534679413], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c86d4522198d1336a1a99ac7ca9c2f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4802755117416382, 0.17946738004684448, 0.5224344730377197, 0.0764428973197937, 0.7114992737770081, 0.6274797916412354], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c04301231200b660557fc2b683c6b415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5152817964553833, 0.3285520672798157, 0.5713917016983032, 0.4692542552947998, 0.36130884289741516, 0.497228741645813], dtype='float32').reshape([6]),
        ]



class PrimitiveOp_906f3f814a87b066634781484c02ba47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1e-10, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9de6f2ac338980c248c1d3386a099cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03883463516831398, 0.00019726960454136133, 0.024178875610232353, 0.002077957382425666, 0.0006825195741839707, 0.02110004797577858], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd41ad5d746fe9dcfa41f5fb49b2d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([0.017001358792185783, 0.11817431449890137, 0.014661362394690514, 0.04215633124113083, 0.03779536485671997, 0.003305346705019474], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7613c035f0612eff05bd3599e31bceda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09524781256914139, 0.1662207990884781, 0.04816321283578873, 0.137531116604805, 0.04644821956753731, 0.1607235223054886], dtype='float32').reshape([6]),
        ]



class PrimitiveOp_3b3f7cc21e270659f61f26eb35b4a9aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.4052850008010864
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7893233c87139d0e8399aed82fa933f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b3f7cc21e270659f61f26eb35b4a9aa
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.232835590839386, 2.3465805053710938, 2.266756534576416, 0.7959005236625671, 1.4818192720413208, 2.3078603744506836], dtype='float32').reshape([6]),
        ]



class PrimitiveOp_c6c2a25318c8711b8df066f6d92e085f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_981fcbb7b7ed0d884998b2e0e8d75992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c2a25318c8711b8df066f6d92e085f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.015826720744371414, 0.0, 0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9030161ff106d4b1c6f5d1c10481d2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0061447620391846, 3.231677532196045, 3.0824294090270996, 1.2567309141159058, 1.8899199962615967, 3.1586368083953857], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_24c16d46633730af30f324c69a86004a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([1.163149118423462, 3.2520620822906494, 2.711258888244629, 1.3589682579040527, 2.232752561569214, 2.4957945346832275], dtype='float32').reshape([6]),
        ]



class PrimitiveOp_a7c5427caa9c558a99961c0f0e44dcb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 10.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6f227d535e1c605e9c5a9d7eb353302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7c5427caa9c558a99961c0f0e44dcb9
    def get_inputs(self):
        return [
            paddle.to_tensor(2.202331066131592, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f67a97e96f64f74af411b7bef53f781d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_294384b8c13ec501ee28a6831118054a
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1adb458e7aac393a8224bc8fa9879ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1adb458e7aac393a8224bc8fa9879ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38790564d46a4252f879950f37aade3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae5463689c5f923bd324104b0a797e9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_315e2795bed0310c2e090be8ec415975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18f46472fe8b09bee004a249dce2c20f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_120afbeeaa68a7c6fe11716b8c458e67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc00a6b8510339ef534e4c72114abfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120afbeeaa68a7c6fe11716b8c458e67
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cc00a6b8510339ef534e4c72114abfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120afbeeaa68a7c6fe11716b8c458e67
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]



class PrimitiveOp_d8fa4d0eef3d4ad9bafed392480baac9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.111109972000122
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b7b3b56daebdcb650dd80210f82c01d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fa4d0eef3d4ad9bafed392480baac9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1658f9b34bcf994e2996ea7eb47bd219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fa4d0eef3d4ad9bafed392480baac9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_43d3db9fee39f86aad292179bb9a6ebf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_318df030f760a7f4ea96ee7f90738861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d3db9fee39f86aad292179bb9a6ebf
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d0eae8051cdd6ee62dadddfd50832ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc921fe7f90e4d97a260f1927eb12b9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0eae8051cdd6ee62dadddfd50832ce0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aeb43ad5b6c4d173e5f8bb2e414ff481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd66f65bc5bc41b742b4ea78073776ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1563340425491333]], [[0.04343500733375549]], [[0.008920107036828995]], [[0.14171743392944336]], [[0.42943236231803894]], [[0.43999648094177246]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d934968e5f29226199191847143ba7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd66f65bc5bc41b742b4ea78073776ac
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.15264664590358734]], [[0.34894150495529175]], [[0.4814743995666504]], [[0.1594444066286087]], [[0.09437808394432068]], [[0.06603606790304184]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0e2e2356d33f239f6d2197e3172913e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce78b0c22c5c73ec83998383f15f05c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.12701515853405]], [[0.04410155490040779]], [[0.37185001373291016]], [[0.14651024341583252]], [[0.4740857779979706]], [[0.1435951292514801]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5dc5101a98ea97a55f09bbe85f8c14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce78b0c22c5c73ec83998383f15f05c7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46422049403190613]], [[0.49471724033355713]], [[0.21156202256679535]], [[0.0383564792573452]], [[0.4534785747528076]], [[0.36828336119651794]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d7e2742e1387d50bc454aa464469833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_38a9311f189da98aedde737b5294222c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dcaa61bc25be87694eeb8e0803f86b7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23327752947807312], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_7ae0d8f715429fca2ca4937873768cd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2.5
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7864d8d6bbe45af7738beefd542c9dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ae0d8f715429fca2ca4937873768cd6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07856619358062744], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_d843957af287465e1f7b6c54ebc20062(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.05000000074505806
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_10183bdef42dae8e340fd2fe148fea66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d843957af287465e1f7b6c54ebc20062
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11238446831703186], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.111109972000122
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b220f43f7336fbb582e0d220c95eb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6f19891320ec9805a2c580934800455b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b66ae5c935a02543ae86b9fed0bf5fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f19891320ec9805a2c580934800455b
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_beea249863ec1c5cb5838843cdb7795f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e02dac541a840de77d59107a6dc71b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beea249863ec1c5cb5838843cdb7795f
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3169fae0dec2069428191fb864f7b550(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b29dcf4646d56b97da83483528287dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3169fae0dec2069428191fb864f7b550
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bd4593281bbb39409ebd4106312e2aac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0fba3d2a22929f3b91e13d6becaa7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd4593281bbb39409ebd4106312e2aac
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_aa4a7a22d1230ab588ec7de66252ec7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb2e87a4c91f0937d147224f0778ad38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4a7a22d1230ab588ec7de66252ec7c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
        ]



class PrimitiveOp_e373cb01384df072bc2d5b2e9125d395(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0cd4d82b5020b8142783f03e81bf01c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e373cb01384df072bc2d5b2e9125d395
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c87fb3e4053e2331f92c24b0e1e7824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_89847e6aebdecffde59630026ae475b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7ec73d924f15c56d602effda96b77de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89847e6aebdecffde59630026ae475b6
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e02dac541a840de77d59107a6dc71b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_beea249863ec1c5cb5838843cdb7795f
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4c1fa51db06b76533df8cf164cd9efff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8f3bb70c9777d541fd25af338181e18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c1fa51db06b76533df8cf164cd9efff
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0fba3d2a22929f3b91e13d6becaa7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd4593281bbb39409ebd4106312e2aac
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6f6e2ca28a9d12e22085538a282f5c79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78c65723d882f59b3e10ae545364e18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f6e2ca28a9d12e22085538a282f5c79
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d683a508c3fe423801eb63ed5613f499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e373cb01384df072bc2d5b2e9125d395
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04613ce5044ff83df77851c8bc2624c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f0719bbc8de2496213e3d45e1a9e377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1639254093170166, 2.2044615745544434, 2.053987741470337, 2.108333110809326, 1.9586763381958008, 2.0109431743621826, 2.1483845710754395, 1.9205474853515625, 2.0875871181488037, 2.137115478515625, 1.9240128993988037, 2.0164082050323486, 2.0322937965393066, 1.9451545476913452, 2.0015172958374023, 2.25301456451416, 2.1551806926727295, 2.0124425888061523, 2.187988519668579, 2.0474181175231934, 2.0394086837768555, 2.271547317504883, 2.2335708141326904, 2.0531158447265625], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8235f7b2c6911ccfe70e41328ff8a510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(2.875631332397461, dtype='float32').reshape([]),
        ]



class PrimitiveOp_ed8e246d07e804d039f097c42d93b9aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_65ca392415f0958b83fa64f580efe69b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed8e246d07e804d039f097c42d93b9aa
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2793837785720825], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_3c23ca2a8490423bc220ca876e64da13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc04813ca55675ebec53f3db07dd2583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c23ca2a8490423bc220ca876e64da13
    def get_inputs(self):
        return [
            paddle.to_tensor([0.250755536471709], dtype='float64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4fe9ecfa90d725c55a7b0728a3e8a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3b259419b07c3cca3c69649b6061c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc327cf986479818994fdc1d913124db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd012b37a24e162e8808b0c06f7efff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76c43c825cc9bb5a214eee54db5dc08d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b296446ad517f5a144939a9ac7134eb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdb0e19e0c319777a0eb14e2aae78b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdb0e19e0c319777a0eb14e2aae78b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bd4f079a3b9abf80ddcc37b0fe45832b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3598fd5f141afcd31e9e5f0bd4615642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6932ef75a92363711b55969bcf5fc78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eac825bab8741d4630bce4beb6c31c69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef095d1d902f03fe20a2bd924d5237fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef095d1d902f03fe20a2bd924d5237fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]



class PrimitiveOp_bf63a070c888da9eea92394a070557ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60d3513e6ff5a14f66b676a71181df9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf63a070c888da9eea92394a070557ba
    def get_inputs(self):
        return [
            paddle.to_tensor([[9]], dtype='int64').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6ef8d9190060dd82cb230ce002b574d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f74b2c6640d727f68081e7375b735ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2430911511182785], [0.23496943712234497]]], dtype='float32').reshape([1, 2, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1582c729164d300befc0f03f18256d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4728992888c5fbfc9aab6d0093d2260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c8d377f245868f63456a833c5d1abd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_574f28f3a6e812f59a875f18f4d2d8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f6296811f6108c3c187c3b10459e4fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1841213703155518, 1.872342824935913, 2.092287063598633, 2.162114143371582], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_798ad2d79dadd5c5125a999c9e4afb37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(0.17350471019744873, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f946b3a9fd9f7f31b5a44b7e48771f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_027e89de75ba9dd57da01cc23bfa7fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef3c25b627b9ae981457d0a950967bf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e2c178f0086b85bdee37dbe284e6c183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(201.42575073242188, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f73ad2c54512b21597eeb1288d43051a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(5.126999855041504, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ddcf873901ad21706700249bfa4fb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(130.0355224609375, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a7eab85d032ce4bc6a7ab91c1bdd31c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.126100540161133, dtype='float32').reshape([]),
        ]



class PrimitiveOp_2925f34c0de139e57583795d4071d089(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.25
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4597c818265165982fb57e458263fa9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2925f34c0de139e57583795d4071d089
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_785eac2191d0cf7fbb6b4c9e47033e56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d457e4d0f614559af45c20bd898a6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_785eac2191d0cf7fbb6b4c9e47033e56
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71b4204aa004edf2e935269fd9c98ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(163.173583984375, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebd84acaac05d2f90fd149b1d76d8bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(9.439167022705078, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0eff9b06d1799db9bd5ddcf25463c99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07083103060722351]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8982d7deb485de824b04c92d7dac5abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.012383874505758286]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ca811d25c49c9ec19afa4c39ba8585d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-6.719618320465088]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2988a239019edf7db297a4f29375be11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.719618320465088]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92d77a627885a2ad898a525e56f9a393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0467146635055542], [-0.060889631509780884], [-0.02047661691904068], [-0.05326319485902786], [9.980075992643833e-05], [-0.037149399518966675]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21ab386e14a8b052d74ea27d170b24b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02797812409698963], [0.019563326612114906], [0.0015397187089547515], [-0.07706371694803238], [0.01333153247833252], [0.021454868838191032]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc8f0a9308dc1518a5353fe5e06d002c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-2.6696853637695312], [-4.1124372482299805], [-14.298932075500488], [-0.30884212255477905], [-0.9925139546394348], [-2.731513738632202]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad00736971b7dfc0c3e82d2131814e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.6696853637695312], [5.1124372482299805], [15.298932075500488], [1.3088421821594238], [1.99251389503479], [3.731513738632202]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df95196ea31816f85e943e7ba797f34e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d21f94561a0c18c769ffc109fc0a4336
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1637033373117447]]], [[[0.5363078713417053]]], [[[0.7413976788520813]]], [[[0.9479138255119324]]], [[[0.3223705291748047]]], [[[0.9293025135993958]]], [[[0.3269089162349701]]], [[[0.5215974450111389]]], [[[0.9874126315116882]]], [[[0.7935583591461182]]], [[[0.5209171175956726]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b49a170875cebe9c908dc33771df5957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41cc2c63c8cec85a926ab2a4e2ebbd8c
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ebfb0095bd4749a11623956387c0b87a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ba4bb42e7a7d50880d548e70147c315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebfb0095bd4749a11623956387c0b87a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
        ]



class PrimitiveOp_8c2bd112336b133aab95d930b87ac918(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9f92044f13da7b6c3ef5451834df54d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c2bd112336b133aab95d930b87ac918
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
        ]



class PrimitiveOp_0ce97bb04975492058c48f5e5124deda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d59944e0905017946b05ebf6005bd424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ce97bb04975492058c48f5e5124deda
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d59944e0905017946b05ebf6005bd424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ce97bb04975492058c48f5e5124deda
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7edf558124bb96d460087cbc21f8d95c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_705294b23ec954e6eaa75ef92d8c524d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7edf558124bb96d460087cbc21f8d95c
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_705294b23ec954e6eaa75ef92d8c524d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7edf558124bb96d460087cbc21f8d95c
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_951127c0f23e6b8ad9acf35e7e98221c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e1a13a0db156429175f741d02f5eb899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_951127c0f23e6b8ad9acf35e7e98221c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
        ]



class PrimitiveOp_bc6a7644ee745700a58c17352e7ea23e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_068423faa4a0f7eb443078760a1e1d13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc6a7644ee745700a58c17352e7ea23e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
        ]



class PrimitiveOp_524c1c0174f259b212bec1a3e34f7c49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4bc091bd3118b1cdc634c882913565a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_524c1c0174f259b212bec1a3e34f7c49
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4bc091bd3118b1cdc634c882913565a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_524c1c0174f259b212bec1a3e34f7c49
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_46f43228328e040cffe01f9370df3755(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc66ec79e5875ad93e2ff7bfec51cb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f43228328e040cffe01f9370df3755
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc66ec79e5875ad93e2ff7bfec51cb98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f43228328e040cffe01f9370df3755
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_399e269243eaa023d3ed14c743db5388(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63f8b4301efe957cbddf8822b6e43262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_399e269243eaa023d3ed14c743db5388
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_17c4268a87b08c6405b9175440b7ccb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2d51c765966543a98e22ac0905e7475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17c4268a87b08c6405b9175440b7ccb2
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_fda26118b540395ec4004cb3a67a0c8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d6d5fe046fe877c5761ea9f02da66af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fda26118b540395ec4004cb3a67a0c8c
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d6d5fe046fe877c5761ea9f02da66af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fda26118b540395ec4004cb3a67a0c8c
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1a45c488d3f35c215338036845f71604(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d43faa1d5aa76967df3a02d2f8969247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45c488d3f35c215338036845f71604
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d43faa1d5aa76967df3a02d2f8969247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a45c488d3f35c215338036845f71604
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19389b2b3dcbeffd7ad11b742e8f54ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80eaefb5be14732f7fdab0a8d543f81c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6081e0fedd7d2290fb7e174e765884e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6081e0fedd7d2290fb7e174e765884e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9e489f29600da1609784b7a5c6ef7a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_650a0a90e4c8aff2c34d752dc5e2d76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f74b2c6640d727f68081e7375b735ce
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24354983866214752]]], dtype='float32').reshape([1, 1, 1]),
        ]



class PrimitiveOp_e49510c8601038319f348a32a601c2ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df980e4beaafc750c2420b79eb9fe231(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e49510c8601038319f348a32a601c2ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_017508ecc884b1da4805ad4be833af2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0015625000232830644
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46ae38bd009df1f60144532fa5485775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017508ecc884b1da4805ad4be833af2d
    def get_inputs(self):
        return [
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7d0a2499e0c3a731bdfec9e15895eef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017508ecc884b1da4805ad4be833af2d
    def get_inputs(self):
        return [
            paddle.to_tensor(7.0, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_034d623f61d5b6e13d27f22fbf41f532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17e45e927e55af3d09c8c4aa64947bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59ddaf42fe225ca800031751ccb16c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(62.63297653198242, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_311f6d1f62e5ba0078e4e95adc0b16ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(2.0307183265686035, dtype='float32').reshape([]),
        ]



class PrimitiveOp_b364cafa88372c92bc076fb1b8beecb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aba259a618643ccdce5cd82f3cffdcee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b364cafa88372c92bc076fb1b8beecb1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddd5e30a73a679891d48f2c1e5ccee00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47f3cf19ab64b58946acaea7546880e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a0dca3c03fc74062cd2e7c65340e3db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b8f444af6da8a57fc6d90828db24d916(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6de984db6953e06fcee06d92aad344c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8f444af6da8a57fc6d90828db24d916
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8b561adc88fd6f915d5e5acc529efff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2fbfbce2bd7381e95c3d53122e5498fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9858c6a4010df30f8773c094f427e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(156.8470001220703, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd568336ff8e7383b457fbf08b3ca2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(65.62250518798828, dtype='float32').reshape([]),
        ]



class PrimitiveOp_0558ef474f3f3c383e42dfae4eee7454(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71f53216ddf82716c8ed732c9ace7b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0558ef474f3f3c383e42dfae4eee7454
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9cf105de78510ce1ed03a167bd72482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a4c6a47120cafe32ca89d6e4e6ba1a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e9c53f4f19af7a00faedbcd8b746df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e9c53f4f19af7a00faedbcd8b746df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06719c8f9604193806e396b18172ef83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c88b00ba0fb5ac642744d7893ab12418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0955e4cc1aa749153519ee36d34da780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b263302192ecb2b0b1015119747e0bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dda8899dd6a817b0183cf7d4e72e77e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dda8899dd6a817b0183cf7d4e72e77e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2484567b7edde4df45b60208539371a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b364cafa88372c92bc076fb1b8beecb1
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_251b02856772b603f655e55bf53d5254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(151.71063232421875, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54b19aa105571f6593c99457ac76ceb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.160825729370117, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f26b7f17bc41e128cc8e583e592d73c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e10d59034017db7df19c153856859ee0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]



class PrimitiveOp_0e61aa038751f1df8d4edcf199bc6d22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45809e1792ce70e6916f01879fd84840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e61aa038751f1df8d4edcf199bc6d22
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2834f464973590a90b3fb6597f28bb54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01693892292678356], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a0b11311b3400a5c069ec5907e7da52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1082133799791336], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_687d5a1e1d600a5770374741809428c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2033243030309677], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01e7a1c109985a61b81c5eaf9c50d775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2884811758995056], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3823db38a71ae9110367d2bc31211d57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1573590785264969], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b45c89f7a9a383ebdd66f6d2b1e22f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5564619302749634], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b46f242a6d674ff613fa0eab8a15ad23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1313057392835617], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3253ba47a206194e32dff365d02c5b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30344873666763306], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da9089d367bd95d21236c65a430fc045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12425953149795532], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f339f768bd065864ff06791b9a8bd09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39257174730300903], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_16c0f67698ce04b63e3b1eb08df9e9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18483184278011322], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c60c78f9169665d13526835593e6d17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33507657051086426], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4101e26ef71b9e1317b8e3f99226b309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2986142933368683], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68e573859d11ced05dd003407d37707c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33657583594322205], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b69b39a395b693d9ff5abfdb48361f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.009820956736803055], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e81b0f59e1e4581201348eb2d0bc8f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.061011020094156265], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cfa6907f25d37c20bfc37edf3b2f10c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_560bac8574ea9e5ee7ee410f1a8aadb1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17482644319534302], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6316c2242466ffc7e6ab3f513d82bd9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45183873176574707], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea0daf5a3d66e6022b2388d0789519ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09283372014760971], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e20514c36ef6090f564b38a33e6bdddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10435719788074493], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2751188c38d47c65532bbefca0da492a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4671162962913513], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad2150e0f25805b684d2fb51ea6bf104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4765782356262207], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7cbb2f2c404ea4d1dead3b01a1edac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7fe9c8f2c4d8857f68188775e865b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7bf06456ba5f1b21eec14a44f51bd9a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aaeee1ea2f00d688a8acdbd928083be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9616412f30553617052bea660ea0f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41582db8f3d96b3a57cf420473e02ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3460f54ba3138460e8e1ff10463a09e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_518d9510ce374553c0c95d7d07a98c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3460f54ba3138460e8e1ff10463a09e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dd68859faa7d41da4fad523b56ef1e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58b7eb7379f4736e8935d32d7d51634d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_518d9510ce374553c0c95d7d07a98c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3460f54ba3138460e8e1ff10463a09e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_130804fa43df087d5fb784dba53d8c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_130804fa43df087d5fb784dba53d8c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1aab39b95fa9ba715d26617dd9156340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9c0747a8178f2db3e880671d4f7cada(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab01eebc2d27df4b6012c6a3e9a69e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ce0b520ab2f8de9c4e630011205b475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f2fbf10ca194b28336c7c2c38448f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f2fbf10ca194b28336c7c2c38448f15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b782107122d84e68620d7bb3b3f7490f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_294384b8c13ec501ee28a6831118054a
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6223eb60e7c2a05de6a6d7544abb59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(98.3559341430664, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b053c75843de399c098ccf1c4d004cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(300.5543212890625, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d8bc95d58b0fdb3499cd9e9e5cabdf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d8bc95d58b0fdb3499cd9e9e5cabdf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e25c25a41bc6a80477e78815115f98c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1685ae26e214a0e9d8b8aed94ff8b10f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da928debb8fb81705699b03561545df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba3dd74ab047fb0f1e72e665967dcece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c90a48287608bd8fa63ba80bc4901c2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c90a48287608bd8fa63ba80bc4901c2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]



class PrimitiveOp_def47833355823697669cfeab831dd55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_388fa05ab80d1a8f94fcada0a4e83b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_def47833355823697669cfeab831dd55
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_58c3dfafa5346b2102ef48bbc3dd2d44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -50.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74ff518ece4646109f1a4ba1bb895248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58c3dfafa5346b2102ef48bbc3dd2d44
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ae6026644ca29cf5aebd9e38f4b8aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cfda4c2d4cac996d001acbe86d75539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(94.67943572998047, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3e4996278c648f3d447503e6ea19d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(3.4263734817504883, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc591d84435facef7cdc14862249527c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_664f9b3b573a9be9c841819852cf4800
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71321b56d31f0b104b7194000c4ebc21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa3109363ce374c82889b389a3686dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5e2da4bd16ae078d6c2ceb27af813ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_73cf776fd1e94655b46c3377445da2d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.75
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec14cf2aff99c930db1fc9ec5cda769f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73cf776fd1e94655b46c3377445da2d3
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37f26ed530297ac6971a933712f7e6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_622e085199e74a32651b79ac5ef475e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1e-08, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07b61a8de40c57ebc7e70d65060b8961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd684732a3c7cdc147490e2264701c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e6ed0646c0443c50b6c8ea44ba4b70ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.25
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c7504e0f7da0795de0d419c1c675a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ed0646c0443c50b6c8ea44ba4b70ee
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07b61a8de40c57ebc7e70d65060b8961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd684732a3c7cdc147490e2264701c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9604c8ff95876414b086347ce35e0e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66ca5a8b0cffd6b5992d71a3e7804c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5c083fc411641973e690bb7c61cd0e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a0ae1176f2375752af8eda2c5861b016(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e0d347cb88c5245755a2567dea384b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a0ae1176f2375752af8eda2c5861b016
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d643e61a1f3d3820999846621d6a698c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06acdf4f5796c09a9e6faee6a58aabbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d643e61a1f3d3820999846621d6a698c
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4ee44e5ad07bc4dcd0efe5f72953ceb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4e0cc24a91a339c78f774cee5675fb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ee44e5ad07bc4dcd0efe5f72953ceb9
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7dc361277317fc49584ef9203b8900d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0feba44ab35027b1c47d6a0d20deeab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc361277317fc49584ef9203b8900d6
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ca1a89766154151cd19cbf371dfd2c96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab9f5bc7dd6e9bf8621580028a2c79c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca1a89766154151cd19cbf371dfd2c96
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
        ]



class PrimitiveOp_004001ce0710028bc672a1d2c87cf80b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b29f14f1584b2101c0a5292f834e72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004001ce0710028bc672a1d2c87cf80b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_828062425350c4087b2ad1c778d4ddc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_55a1cde715466b603d6d1eecce315a85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cdd3f966832b4c8b83fe276dc3942a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a1cde715466b603d6d1eecce315a85
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06acdf4f5796c09a9e6faee6a58aabbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d643e61a1f3d3820999846621d6a698c
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_db997e2a32f138aa537055e5cc84ccf4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_728c026c32ee3d0e7ecd3796fa94819a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db997e2a32f138aa537055e5cc84ccf4
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0feba44ab35027b1c47d6a0d20deeab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dc361277317fc49584ef9203b8900d6
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2047b7a8b5bc1def86c53e7af09e7dcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d9b003cb5b5d3f24cf481ac7b642da4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2047b7a8b5bc1def86c53e7af09e7dcc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_315257d7efdf62c6295a138fa9e23046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_004001ce0710028bc672a1d2c87cf80b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a072b0448d8bb76bc15cb65daeec27b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a072b0448d8bb76bc15cb65daeec27b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a072b0448d8bb76bc15cb65daeec27b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_82f9183f7c872e9e5d58be275aaebed6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.111109972000122
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3353beb55f191c7b66904e202ad5986a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82f9183f7c872e9e5d58be275aaebed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_272942409ea561f3f00f4e7afffb6d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73cf776fd1e94655b46c3377445da2d3
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5c6237b36e9ba428c543c50c02d49fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538025fe57de5047eb05d808f78021d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52400e5229c03e51d71877a026e857d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf7d0c12bb9357fedb433b365f5956bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ed0646c0443c50b6c8ea44ba4b70ee
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538025fe57de5047eb05d808f78021d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52400e5229c03e51d71877a026e857d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17c1177690e6263ad4e584dca8710357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07234402745962143], [-0.048308003693819046], [-0.026102447882294655], [-0.060440488159656525], [0.03520594909787178]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cf9608c70b472788dfe5ce3a0a5d070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.025636205449700356], [0.07950747758150101], [0.0040383669547736645], [0.08718772977590561], [0.03009037673473358]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4541d507325f836d5fd92efa01b204a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.821947693824768], [-1.6075907945632935], [-7.4636149406433105], [-1.6932224035263062], [0.1700069159269333]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9dee8d39fd9eadcdcf9accc51e0f89da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.8219476938247681], [2.607590675354004], [8.463615417480469], [2.6932225227355957], [0.8299930691719055]], dtype='float32').reshape([5, 1]),
        ]



class PrimitiveOp_ffd0bbf593198de51e87726cb7f3a935(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3363712c3b1621f6f64e37978f7b6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffd0bbf593198de51e87726cb7f3a935
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b224087c01ad6d103c153c2f7172338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]



class PrimitiveOp_bf2849a29c533bbc9eab239a6da17c19(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 64.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93e5217e49ef45f509acff8a6e0fe6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf2849a29c533bbc9eab239a6da17c19
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b224087c01ad6d103c153c2f7172338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93e5217e49ef45f509acff8a6e0fe6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf2849a29c533bbc9eab239a6da17c19
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e5f31a2f88c39611800cf5b3e6ba1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
        ]



class PrimitiveOp_ec37bee4e48a95c86d20024ad5eaf317(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 128.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09308fc862b2a19261dec7942f8f0f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec37bee4e48a95c86d20024ad5eaf317
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e5f31a2f88c39611800cf5b3e6ba1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09308fc862b2a19261dec7942f8f0f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec37bee4e48a95c86d20024ad5eaf317
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
        ]



class PrimitiveOp_a9596f395d20ffb885c6a537f30fc4c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5683c4df2cae1fdef8a583e32e63650d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9596f395d20ffb885c6a537f30fc4c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae7f05ccc9196316003a178ce4b4cc83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99736c3227e24a8eb22968054a4c1884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99736c3227e24a8eb22968054a4c1884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99736c3227e24a8eb22968054a4c1884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b7c53b6f7c3efca5b4397465d0fa7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82f9183f7c872e9e5d58be275aaebed6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_327477cbc07eeee8227015f66f3b3040(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71cdb87fcfc429cab1badeb674083d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_327477cbc07eeee8227015f66f3b3040
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e56642c4fc165a4b7395af89c080f6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58b7eb7379f4736e8935d32d7d51634d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71cdb87fcfc429cab1badeb674083d59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_327477cbc07eeee8227015f66f3b3040
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3d249d43fe80355bed1630c616d284b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2480415f0c799c9bf7ab726729d6750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ebc127f2cc30fe443dff87b2144036a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e5c222b6a187ee0a0a552b280d74163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58dcaf5d14ec113856aa372c89bf1b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_849157daa5639426c4c5a3dba4dd79dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b2f193c3a6ede418d4aa277c5bef4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2853190004825592], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8852e9c12ee80601291401dab1a27d48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ae0d8f715429fca2ca4937873768cd6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19351647794246674], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f2c079d3f264d1b03949eab68825635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04894401505589485], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_24c30e556a8b76544d016099510e6a0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44e8a38873aea2db72931fe6f7c4daa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367d1f6aec56641187e36666d5804fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aa7b7d93b00ca94675f8db2b6b1bd2a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa877fdc839c3d9af416ed8f2baeffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3dba9952288c1f6e1973cc2a60d2d340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3dba9952288c1f6e1973cc2a60d2d340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6b2bf8778062ec136d28231138d5042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5ddab3de0eb470ac4b0bd7ed8ba784d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6579b8627ee481a1c449cac48a3220a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a9c99fe648c46640ca42f621cb2d978b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5345a9e405a2150b7e0c0afff513febc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5345a9e405a2150b7e0c0afff513febc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec16772a309887af8212e2473157280b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec16772a309887af8212e2473157280b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41d84ce44da00c2286f04e62c6b4204d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a36a45b7c4265c5842acf2328d0cc256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_897a8c5f88202e792bec3e606269c82b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5196e6f2a99a002d82a336b24257b870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e2223a4cec76174e4c3b35e6c53a582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e2223a4cec76174e4c3b35e6c53a582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddcb2df3d139c2f31794c6b221efa8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddcb2df3d139c2f31794c6b221efa8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0deb2b91f3028fa485c49dcf046f189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f33a50cc334c98ff76732fa6d0feab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8aa7f05c11b3675988d79113f578198e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e2438adc5dfac8077ef16c20b46194cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc478876535cce8f6b1b81da5faea188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc478876535cce8f6b1b81da5faea188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]



class PrimitiveOp_b9377d1a6be3b1ecc3b3c8f6f5dc2f0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 64.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddb8ac0e034508d0740d5df113df8f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9377d1a6be3b1ecc3b3c8f6f5dc2f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6de715376bcc954853a215b955fe7f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58b7eb7379f4736e8935d32d7d51634d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddb8ac0e034508d0740d5df113df8f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9377d1a6be3b1ecc3b3c8f6f5dc2f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9de89a4432864df9169936136d168384(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9ffa76f0ed5efef35cdbda0d778f736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_233cf009bdf5070e5a30f77f142512f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_525d7b99749653c47e3c30edc53fe51d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.925, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2deedbc4d7785ed1d0a471e64cd67c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_525d7b99749653c47e3c30edc53fe51d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5860388875007629]]], [[[0.8120017051696777]]], [[[0.9539660811424255]]], [[[0.8134338855743408]]], [[[0.6751497983932495]]], [[[0.6726080179214478]]], [[[0.27094537019729614]]], [[[0.2922995686531067]]], [[[0.2422485202550888]]], [[[0.689205527305603]]], [[[0.37748992443084717]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]



class PrimitiveOp_443507fdb20b14fefeed764e57c7e047(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0810799598693848
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ae556f75ee7e2227d7c181f4fe5c940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_443507fdb20b14fefeed764e57c7e047
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7cbb2f2c404ea4d1dead3b01a1edac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_956719df85894c04c8e216c6a25355cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_27282b6c042c9d7955da67c3d4d399af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21808d3ec2c86f8d5402498d65b86b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27282b6c042c9d7955da67c3d4d399af
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_521ac95902a29e7dc803f82c30443ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8ac4536e4ff567573fa673c8bae5897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7c0770325b9bf8ee604992a042881dca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_706d325a4fcce46a8b90c4c7ea9ba772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c0770325b9bf8ee604992a042881dca
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_170f2ebe6225119df88464af864b96ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ce4b7ba7b6eb4a6b9051a6b315a650e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_170f2ebe6225119df88464af864b96ea
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2c19066007ac90cdc4ecb79035154644(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b010fe21a875297322f123cf7609120f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c19066007ac90cdc4ecb79035154644
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ca122d8ae2da08445d3bff945edcc3af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b6f1486f3771976085ba2303f60c1dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca122d8ae2da08445d3bff945edcc3af
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bfc695835052dbc668aae1ab02aa25bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b2141170ff56a2db731f503ad6da655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc695835052dbc668aae1ab02aa25bd
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b2141170ff56a2db731f503ad6da655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfc695835052dbc668aae1ab02aa25bd
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2ebb6ee48eb47b26abdcdcf42c959ef6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11b364d611bcf8738455eced03675813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ebb6ee48eb47b26abdcdcf42c959ef6
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11b364d611bcf8738455eced03675813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ebb6ee48eb47b26abdcdcf42c959ef6
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2f5de92b455ae2b5e9ec7bf359fa5009(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0afffd33820a5a4520b9039bfcfb251b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f5de92b455ae2b5e9ec7bf359fa5009
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b59b210828507b8f45902a06b77d1bf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91b549ac14c04cf27a36b9349598814b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b59b210828507b8f45902a06b77d1bf7
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_62d6d193452029e3ba6635e6d21a825a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0767d289a0b3c31a55ee6458329adfce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62d6d193452029e3ba6635e6d21a825a
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_67d2126f05baef0809d9b34541087be0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5eaa0f57c2c0e8bf1dd60637455dc452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d2126f05baef0809d9b34541087be0
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_37741b151164c46a1c4c650bef0d4db9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f7a1f78235eecd3bc401a2f93618f21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37741b151164c46a1c4c650bef0d4db9
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f7a1f78235eecd3bc401a2f93618f21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37741b151164c46a1c4c650bef0d4db9
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_46668e25dd3cb67c3cc0cfba05abe165(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b666c6b81f1686ad91d8b61eec04d1c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46668e25dd3cb67c3cc0cfba05abe165
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b666c6b81f1686ad91d8b61eec04d1c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46668e25dd3cb67c3cc0cfba05abe165
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8b543134c7e0a8da99f420a0378ca7d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_216dc05e2123517cb51826f8d3214374(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b543134c7e0a8da99f420a0378ca7d3
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_389c2b7a00e21e4a58ab21d2caf0cab4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a604dd0979f091877a0f04c495146ca4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_389c2b7a00e21e4a58ab21d2caf0cab4
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5a27dced28184c7809b569ae39039eb5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1417af72411ea31b9775217052240c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a27dced28184c7809b569ae39039eb5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
        ]



class PrimitiveOp_569595f131c47d3da80f919c206b2d6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e9cf8bae55e21ba40ee600cdb03edd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_569595f131c47d3da80f919c206b2d6a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
        ]



class PrimitiveOp_d9868eb506b306e8b464372d5dc99395(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_544553fbbde038b093d1e3a4957d1901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9868eb506b306e8b464372d5dc99395
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_544553fbbde038b093d1e3a4957d1901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9868eb506b306e8b464372d5dc99395
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0bded33a2992cd5c553a77bb8ffb35b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2eaabf3d1359dc1eaf6f680f2b5fe2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bded33a2992cd5c553a77bb8ffb35b4
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2eaabf3d1359dc1eaf6f680f2b5fe2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bded33a2992cd5c553a77bb8ffb35b4
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_52d9beae3c99674936ea0fddc52241f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c33b6728b60793aaa21df7576e7c83f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52d9beae3c99674936ea0fddc52241f5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
        ]



class PrimitiveOp_539016cd8d1e90eef0ab529aab2394a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 64.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4245128fe33f12be30c4eb6a487e0845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_539016cd8d1e90eef0ab529aab2394a6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
        ]



class PrimitiveOp_37bdb6bbc2a1cf0a16282a833b88c264(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b1dcaa6bbb0e5b33de944f77f5257622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37bdb6bbc2a1cf0a16282a833b88c264
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
        ]



class PrimitiveOp_e71f1aefa71c232222114178d4a92d0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 64.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_35c0686f57298356e1dfad00e8f80a00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e71f1aefa71c232222114178d4a92d0a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
        ]



class PrimitiveOp_2c75637595d73b94088cec5a18b00a86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c73dd303c92c4439c3677896a8138abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c75637595d73b94088cec5a18b00a86
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c73dd303c92c4439c3677896a8138abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c75637595d73b94088cec5a18b00a86
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_257c899679c78694a7295e1846e91ed9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18d43d7691f9fd0cfc4a6c45a6d722b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_257c899679c78694a7295e1846e91ed9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18d43d7691f9fd0cfc4a6c45a6d722b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_257c899679c78694a7295e1846e91ed9
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4330bc7229c9d642d6e2e530d89c623b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b541c3295ff17d7f5a1a4a6746338d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4330bc7229c9d642d6e2e530d89c623b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
        ]



class PrimitiveOp_67231e852ecafed716bd4f41ea158246(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 128.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05571311a6e2ff3cb1f605f67f1b6ec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67231e852ecafed716bd4f41ea158246
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
        ]



class PrimitiveOp_f6e9a84a5ca26fcfc011e79002f3539b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d127172d14ef1c046ad17332628b1b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6e9a84a5ca26fcfc011e79002f3539b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
        ]



class PrimitiveOp_1e370b606bd6f549f099c383373aeef7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 128.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64e8046d64581e15229250e7cc2faddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e370b606bd6f549f099c383373aeef7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
        ]



class PrimitiveOp_7481caf43fa7aa856bd2510438ffddef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b7086ffe29f924bd8b57bc360508ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7481caf43fa7aa856bd2510438ffddef
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b7086ffe29f924bd8b57bc360508ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7481caf43fa7aa856bd2510438ffddef
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_80bae86b4d6f193806ad76012f3a8b7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1fa849d2e83acef15fb331ebfd13137a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80bae86b4d6f193806ad76012f3a8b7f
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1fa849d2e83acef15fb331ebfd13137a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80bae86b4d6f193806ad76012f3a8b7f
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b2cd29ca129a96acc9ec0b0575749a73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.111109972000122
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bfa0e4cf7e233f6a31140a4f1f64d3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2cd29ca129a96acc9ec0b0575749a73
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e2cea0778dd0f0b332e78af418f9e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fa4d0eef3d4ad9bafed392480baac9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7c5244e75aaffe719d961ffcd5d284f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a37c8121f1b45eec87d03f896d28dbd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.116051197052002, 2.2622504234313965, 1.9536495208740234, 2.0681753158569336, 2.0806095600128174, 2.11643123626709, 2.0751419067382812, 2.1351099014282227, 2.1985647678375244, 2.044846534729004, 2.0799427032470703, 1.9587132930755615, 2.056234359741211, 1.8934751749038696, 2.1239142417907715, 1.9864888191223145, 2.1752965450286865, 1.9954736232757568, 2.161653518676758, 2.157972812652588], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_736c06a2bee6ce7510a1ba7747b15882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(2.5518760681152344, dtype='float32').reshape([]),
        ]



class PrimitiveOp_b97cef8fee74b2db32fe28636007766b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de1ec10d8aac60bf5f8124c6e1574a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97cef8fee74b2db32fe28636007766b
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bd60325cd0c36f6370722a7590af1e05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5e79e976acdfbf53ac941b6b92cec0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd60325cd0c36f6370722a7590af1e05
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17540281007f73c6237047303b960486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(366.6561584472656, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2690f6458fe89cc1aeaf9560fe2c330e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27282b6c042c9d7955da67c3d4d399af
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b8f45839b32845f8f66671f8e3f865bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.006892362609505653], [-0.06635904312133789], [0.016340315341949463], [0.024487247690558434]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a5b3ac1c1b89a7f723c2cc61a062ee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.022870756685733795], [0.04407133534550667], [0.00013807692448608577], [0.05499931797385216]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99aaaa30c051658c056794531c7276a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.301361322402954], [-2.505718946456909], [117.34202575683594], [-0.5547717809677124]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7346d950bb89736c4ab0569783739468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.301361322402954], [3.505718946456909], [-116.34202575683594], [1.5547717809677124]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf1dec75865eaf7a5b5f53d364f52fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_334576c0905dfb146a57147f41aa83fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(33.52735137939453, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5f946428fdcce9fa74f5c68cd144faa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffaa6fdaa22f75a41a46021c19df9c57
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44230620cfbf1d420a1008f5e55b740e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bbc8818a02fa79df132b41b02f5557f
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0162105d5b5e6bb3d641e9d45abf90f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0162105d5b5e6bb3d641e9d45abf90f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b715f61f35edb180c494586cbd2b7331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8246c50a39127ec01a49d6a71cd69ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e2fa435619dc56371c2886aa329c437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8163fe10a7fd0ad2bf604b49a98bb60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b208d5b3e143cde62d63537a6a3bbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b208d5b3e143cde62d63537a6a3bbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]



class PrimitiveOp_714e2584365307c87940e8861ee91155(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64e8ef7304a603975aba1cecedc758e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_714e2584365307c87940e8861ee91155
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0640b9287a25dc9be159ec6ea3a94d40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e9cf6b24b17c67a2e0cc0157aa06005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0640b9287a25dc9be159ec6ea3a94d40
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f45f3ef3c9606a8a3bb6527e47622af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_717f9dcfa09d0252cfac6192ff52bbfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8f444af6da8a57fc6d90828db24d916
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8d3ac68dc8dd36a266847354cfa1543f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_384c9061d4737f54ec17a30997c63cb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3ac68dc8dd36a266847354cfa1543f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1b9c251c6d55f61a777b1582c1b145dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b0372de235617a57c104668a2c30e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b9c251c6d55f61a777b1582c1b145dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_884f4c8cf3d0b304e9442ce7fb35d7ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58b7eb7379f4736e8935d32d7d51634d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4b0372de235617a57c104668a2c30e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b9c251c6d55f61a777b1582c1b145dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8bed7d1c8177c8b1a547021d7005e6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e31d58a844c105b55f981ae123bf934c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fad4f214604305e3f5baf9e73a5af9cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6ac0142b60fd3504e9ae77a4ee402312(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2b7c789e7bf4038946c660009b3891c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ac0142b60fd3504e9ae77a4ee402312
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_465e7cd84b3db3b3a286dd838beb6764(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1193053c5a45c3221f9437dbd63ea074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_465e7cd84b3db3b3a286dd838beb6764
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_cecaccb49ba62ff32db0d2bb7b25ff93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_36861dd52b872af3259fef101ae397f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cecaccb49ba62ff32db0d2bb7b25ff93
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2ee18eb46edffd395eccf54b7c58a21c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb11715ef9548ae56cdb21eca7d1e4de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ee18eb46edffd395eccf54b7c58a21c
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1e1854020b265db662d206b8b417c09b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55c6a09bd8d956a6a3a451d0be1e25e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e1854020b265db662d206b8b417c09b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
        ]



class PrimitiveOp_ebd7ac16adf4ccd00196080cb3105e01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4c0a1785c9e0e159f54b80a28206330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebd7ac16adf4ccd00196080cb3105e01
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dac75af7fd0e03ffe04021ea6de21a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_398ed2dba836aca0bd50ddf53ef9f1af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bb0b5a18ddc948db2cfd041909edc841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_398ed2dba836aca0bd50ddf53ef9f1af
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1193053c5a45c3221f9437dbd63ea074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_465e7cd84b3db3b3a286dd838beb6764
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ee94a4634d6f348afb09db3fbfdce939(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4c84cf33b615db58be7c1f8a40dfb61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee94a4634d6f348afb09db3fbfdce939
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb11715ef9548ae56cdb21eca7d1e4de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ee18eb46edffd395eccf54b7c58a21c
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_641d6725681fdd7d9d67f69936e171b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.5, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e79f4c3f4659ff5977c03c15f75a5ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_641d6725681fdd7d9d67f69936e171b4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de41000c46aa90dc1c291ca2a446c31d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebd7ac16adf4ccd00196080cb3105e01
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f38d285fc84096200097eb6bbef8c6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(92.70454406738281, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fef9726a7cf05058728d1e5735c69b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(1.8303749561309814, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58dcaf5d14ec113856aa372c89bf1b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_590afcbd8d4ed4bf2211bc383f8fa29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a092c4eddcddf77e83403f110ce2b2c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ad1e43f77d23895664229044e539605b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4df5eb01e2a74b2baed3d3285d7b7ce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad1e43f77d23895664229044e539605b
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ef9446bc540b9f57362de5e351c29a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_785eac2191d0cf7fbb6b4c9e47033e56
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aee5d09102bb8b1d9fdb9607c7600f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99736c3227e24a8eb22968054a4c1884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698c9603cf85282204f973df84f4df7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_346c04be3fb09e2ee5516f37e1989676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b220f43f7336fbb582e0d220c95eb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39fca2bebd4243f221f8a3f0503b878d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c425c4fc58d705f167c9086bf7833522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c6662722abeaf266d14fd1d5a4d865c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_700fec5af778fb2dbab2aa06d008a408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(37.852294921875, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f7aff5db603a6c2e7fa0bac0d8cc94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f35501f608ef1621d0f640c56074e83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3830331601fc0057ca126af68a47ab89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3830331601fc0057ca126af68a47ab89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf84f6da1d3bba2b77972aafb3cec4ad
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_98425ee31826bcae2e3dad01582912e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55a3abfc46ba537f89e249d4184d8c0b
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1421cd2b2d1c6b27c407c58397e2100f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e7dc795f599c621eb668daf79c56bd5
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89e748f648a725c5f9ce79924590ecbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2d288c654765f4b64d37732648cc700
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08c5f49fdd66bf43f1e73edf5b57d223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdd2a2e8cb4bb9f2d5471eb58a61c6ae
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4eec34b782c80bfc5eba548f9ccb56c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4eec34b782c80bfc5eba548f9ccb56c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f91cc210d754d4872f7008fbd4159a6b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_08d98c7429f8d07c422849edaa66ed10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d44e8a38873aea2db72931fe6f7c4daa
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_dfb8ee2e5651ea55614900ba35636029(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3b573e41fa2e8c0fbb7b166c892e292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dfb8ee2e5651ea55614900ba35636029
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f36eae1028c702a1df0ddf15af28fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(191.19805908203125, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52da9517138f39d5df9e475445482aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.286322593688965, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc6e281e0dc741474308816b9d2bfba4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad1e43f77d23895664229044e539605b
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f7aff5db603a6c2e7fa0bac0d8cc94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_383600e57e743bd3fffb65d12328aa24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_35a3763bba0380cbb1f9994b676df33c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ed006e1d24a94bc42198c176f109b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(36.31705856323242, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3458a6070a1c3bc46419c696b851044a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_088fa044ab700af81b3be135f754d3ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.17677700519561768
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6cf706273c519442a8e69524121bdb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_526eacee885915fceaa55e37399e5a90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.125
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b602ce1b7c8c602885adbf24dead96a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_549241dbba082ead0f18dab8cc685b2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.85, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_62bae46ce7542d1fa4421c20381899e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_549241dbba082ead0f18dab8cc685b2b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.2914212942123413]]], [[[0.39375486969947815]]], [[[0.16414831578731537]]], [[[0.9359673261642456]]], [[[0.29178929328918457]]], [[[0.3484293222427368]]], [[[0.662407636642456]]], [[[0.027838630601763725]]], [[[0.07953529804944992]]], [[[0.027762018144130707]]], [[[0.6288183331489563]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f44c7c5133a47ebd203a168c4cdd876(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4996152f05746018494ad632380a445a
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_df2291caaa89ad351d901fdfdd9d9045(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.875, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_97223077bba5d57be6d5c7f6ff903ab8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2291caaa89ad351d901fdfdd9d9045
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_658fc5a2490bcc91dadc884755c56347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_947e9bf7f3561d180d896d73968e26fd
    def get_inputs(self):
        return [
            paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72eced3457b18cc1b955150fd7ff392c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.2106356620788574]], [[1.2134649753570557]], [[1.5126121044158936]], [[1.2061398029327393]], [[1.3825193643569946]], [[1.2376539707183838]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd74bc5b188372ed2a29a5d91fb5df65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.393440842628479]], [[1.0911762714385986]], [[1.548488974571228]], [[1.0763018131256104]], [[1.0530097484588623]], [[1.4913479089736938]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92744003fcfc504433a5f6c55c01db79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_749f0500f667bdf6c86747987f6fc03d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_692787bef9edd3d8d3323117c846490c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca9b7b238952692baef0209087276d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8af85bf8446243c270dead072fecf1c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.10000000149011612
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68823cb229d3241897158bd9ac309235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8af85bf8446243c270dead072fecf1c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3268720507621765]], [[0.1784019023180008]], [[0.18456639349460602]], [[0.1489272266626358]], [[0.420693039894104]], [[0.3033420443534851]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a87c867438727534dfada97ed15126d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8af85bf8446243c270dead072fecf1c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.08347339183092117]], [[0.13196611404418945]], [[0.09201230108737946]], [[0.3147572875022888]], [[0.2593747079372406]], [[0.21239152550697327]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_a9cf63cb73d4b66f144bde656f22adaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.20000000298023224
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fa46cc64eab48ed7e8cae02f2e5b3bdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9cf63cb73d4b66f144bde656f22adaf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0281860139220953]], [[0.44767507910728455]], [[0.08558890223503113]], [[0.2602667808532715]], [[0.10315723717212677]], [[0.27075910568237305]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_315831fe798c63adacacf7bb859fb65a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9cf63cb73d4b66f144bde656f22adaf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.45546549558639526]], [[0.3317403793334961]], [[0.34772875905036926]], [[0.4843132495880127]], [[0.25377902388572693]], [[0.41150492429733276]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b38245720d0d41244176aee139ab2674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b337c9a5250c040cae55bb1d1ff37d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc2ad2a3179e747859ceec88863864d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f946b3a9fd9f7f31b5a44b7e48771f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_040f39aaddf887ddff3436a738e833a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab7e67481c1d26daf83ff4b6d5879155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.to_tensor([[1]], dtype='int64').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_634493093ddcc1d1a39ee56347153b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2437746524810791]]], dtype='float32').reshape([1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_214a87bc9823ae0e4f99f421c8212707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_819fa67ec5dc8d066f1895000f2d5875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d0a6ef13c05a14b12c2695fa68cdccea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a657de21b19f831a4eda06e013f428c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1799e8a9e09e0deab9e39e6ee49b913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67be5663ec43441078e769e184b3a78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d653045222e7f18b90418c4cb5339429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_352b3a8745e12d705baf529c18540ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6aebcc95931c7efd5c7e61b3b3ef07c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2b63c694f540ca74d88d798d7a50165b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c4537817b9df823a8958a5f12093779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1799e8a9e09e0deab9e39e6ee49b913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7509f1a149f1535242db5a4ff53d813b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d653045222e7f18b90418c4cb5339429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28cff3e60a2d5a8e6b1303f77cff3710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f30a439d4172f0a00fdce6ff5c62156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8b561adc88fd6f915d5e5acc529efff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2eb41fa673c7a5c9e17465e8c3fd002c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e7e87107cad0b026d58a20985570358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4fe9ecfa90d725c55a7b0728a3e8a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66c675e75b3239325aa3771ad588f437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_64bffea55a37388ce26c0bcd427efad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(1086.693603515625, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd9c2163fb338ecfcb122a4fa0641997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(180.47015380859375, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44abe3fe6d4864f8e914e4d46a6e95b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(5.625393867492676, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d63880fa44727551d52cc013b81c78ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.04387051612138748], [0.006444363389164209], [0.0013840901665389538], [0.002383660990744829], [0.010043813847005367], [0.012843223288655281]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9176c689fe1a42673312857add8ad927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.0013636639341711998], [0.00024246015527751297], [1.1780663271565572e-06], [0.0015065169427543879], [0.02632260136306286], [0.0008185577462427318]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b7cb9a92581d925c2df0d8e1c99cdc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6b5b27ef4cd57092c72a992645f6099
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_12637dbd5d01a42ee042cfe0e7d6c453(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.08333329856395721
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aea3dc0f046f3152bd50c1809605e9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12637dbd5d01a42ee042cfe0e7d6c453
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.4026823937892914], [0.07231664657592773], [0.011129551567137241], [0.07582680135965347], [0.40325236320495605], [0.13427451252937317]]], dtype='float32').reshape([1, 6, 1]),
        ]



class PrimitiveOp_89e513045854364d6d4044f81413e968(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 6.28318977355957
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2adf528d7942ffeb19258423e6619c9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e513045854364d6d4044f81413e968
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.033556852489709854], [0.006026384886354208], [0.0009274622425436974], [0.006318897474557161], [0.033604349941015244], [0.011189538054168224]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4034576b0057f91bd90d0a4130837d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.21084406971931458], [0.03786491975188255], [0.005827421322464943], [0.03970283269882202], [0.21114251017570496], [0.07030598819255829]]], dtype='float32').reshape([1, 6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa3109363ce374c82889b389a3686dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_672b1db58f1b08587df4e69f739a36cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6675bd164c9df9618cddca418e6d69c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b2d3dbf6d80db338258d056c3348c25a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.9125, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4e078b37c7ee2231bcd0477a33262a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d3dbf6d80db338258d056c3348c25a
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59e45c172d8d69bbee207ad02990be93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3065ca0207c6c11804a522f0c71ea94a
    def get_inputs(self):
        return [
            paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b1c750136271e02015adf99b0be3a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea873f1632c418d73a1421ff626c3b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9f670c5b148606fccae5c3428d23ee1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1677451133728027, 2.1457746028900146, 1.983431100845337, 2.2028181552886963, 2.1208109855651855, 2.1369168758392334, 1.9550294876098633, 1.8417924642562866, 1.855881929397583, 2.023456573486328, 2.1109743118286133, 2.241074562072754, 1.9125232696533203, 1.9445034265518188, 2.164375066757202, 2.1581637859344482], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05a03ce5a86b3b5b4b76a6fd7e377a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(1.9255192279815674, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1582c729164d300befc0f03f18256d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eecc743746b943c12d4cf3f61e97474b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37653ce0017f32bed764f94d47619b92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_036f3f6b205600ba596661f4e0012a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_036f3f6b205600ba596661f4e0012a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea1950c50bcd7bb61be0f9c98149530a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92744003fcfc504433a5f6c55c01db79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_395ca1124158bb0c79682d510448e780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_263d9b3686b6d2e6cd254ca409f29bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39fca2bebd4243f221f8a3f0503b878d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_148fcaa9483982d88458c05fff486637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68dd3720e439087864c85e52e5305bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(33.84434509277344, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_599819bec7c36992be89447b66dcbe5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8d10799fbc399eee45dc775e493f1be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2abe6d886794017b4873f0c2cb24b818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7785f31dfba0bf596e4adf759b14dfe1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_97cb3602b7321ae480e907496b902331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c11ff0c29dd14d76cce66a10bb05cf29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3d249d43fe80355bed1630c616d284b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2bc635de2b2b6d279db476288decf4b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e13a861f5707de200b1131006cda5013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51a50e644625150c669764d35a5c2471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_25e13a00f2740c98ec2c052e89700459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_25e13a00f2740c98ec2c052e89700459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_408679fa8c61de0c99db8bc7e1b46291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ee5d735d29852902c293c308523e0806(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dad7bd9e909323c211bfaae33589cc92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([1777, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_05bd670fe3209a01015ba98799305f3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2bb6a8a30cd4857b5acc272cea77efc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4a55c5089745d126e124228380f08d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf93e6a40908e3c17823185e7aeac015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf93e6a40908e3c17823185e7aeac015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d7e2742e1387d50bc454aa464469833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bbb92d2ffde1ea806cb404abd81e7555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90051e44ed8795c025505e2b43a4f872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df2291caaa89ad351d901fdfdd9d9045
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8137500286102295]]], [[[0.23146536946296692]]], [[[0.6546924710273743]]], [[[0.5924586653709412]]], [[[0.720824658870697]]], [[[0.24277272820472717]]], [[[0.4453102946281433]]], [[[0.6353831887245178]]], [[[0.4441324770450592]]], [[[0.6455785632133484]]], [[[0.2848832905292511]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ac044c2fd080bdbeb2615e855eb8982d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_947e9bf7f3561d180d896d73968e26fd
    def get_inputs(self):
        return [
            paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_962946b81397f706108d80fddc5f1950(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.95, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_627081581a48ef6c7e62ae7ca9a1b201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_962946b81397f706108d80fddc5f1950
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.001793922041542828]]], [[[0.24888120591640472]]], [[[0.15599043667316437]]], [[[0.9758208394050598]]], [[[0.9929758310317993]]], [[[0.08918146789073944]]], [[[0.9301931858062744]]], [[[0.4761162996292114]]], [[[0.3519049882888794]]], [[[0.513953447341919]]], [[[0.2637135684490204]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fac9c66ac37139dd16015218df8ae4ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bbc8818a02fa79df132b41b02f5557f
    def get_inputs(self):
        return [
            paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fe18f846286e29dc9291fa4760773b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aa7b7d93b00ca94675f8db2b6b1bd2a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd3cf90a1a9db370da0a320ff2f252cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0b2f40668ded1f14ad11532b48914c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(110.48657989501953, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a651ab8f784078f2a60c7482d4faec9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(3.6162848472595215, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddd5e30a73a679891d48f2c1e5ccee00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7cc2db59c68fbc3617f87b7b39e0c5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd012b37a24e162e8808b0c06f7efff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3a1bc86bc5f58f28a870cebef32b0256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_30f2f6cd9872451c57909ed349da4936(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e2169bc594cf37a102ec8377183232d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8b4933ba91b26a287502e2d0b806d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75c0a61e8d5a794f3d7f1b2a5f59e647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecf265cb4d7327c35c8c800a4a3b2b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0c8430262423984451fff540fd3449e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d8d5d1e0dd10499c169c17fdbee105a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77747db0509fc2944c1be95b1d1dc40c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a8b4933ba91b26a287502e2d0b806d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecf265cb4d7327c35c8c800a4a3b2b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b224087c01ad6d103c153c2f7172338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_69d316349753f84673ad988f6da78ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17c57f0ec6c46d2af1e4054b4cae0a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7785f31dfba0bf596e4adf759b14dfe1
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3fa83374d765ac4c4457bfe02d9fdb1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76ed9045f2b3f75632f967c05d2a3cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(182.99789428710938, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_35fd57c66ada64b4069ec7451cbe2165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(3.0948688983917236, dtype='float32').reshape([]),
        ]



class PrimitiveOp_7782fedb136aa160b554a62eba14c90d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.975, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3145d3495ea3ab40f7c46d5d7966c71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7782fedb136aa160b554a62eba14c90d
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f6cf0d15be24daa0af110896dd22dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41cc2c63c8cec85a926ab2a4e2ebbd8c
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca9b7b238952692baef0209087276d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9604c8ff95876414b086347ce35e0e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81b68d432d25287aaa42129b90be1605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ebc127f2cc30fe443dff87b2144036a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d0b2784c43ecdb9b46647c379e3aabe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_149306fa7f6a30a6de00702b100b8b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3495fb43ea9d40660329e055376762a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_612c23b2bdbfd224d9b0af3a8fc80846
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_13b835ddec87a75bc248bfd9206ade56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.023871352896094322], [0.0072302743792533875], [0.028566552326083183], [-0.012393927201628685], [-0.014890296384692192], [-0.004542335867881775], [-0.037730999290943146], [0.046309053897857666], [0.10882711410522461]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a5cad223a778bf47060e386961af8e95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0006073885597288609], [-0.018595393747091293], [0.022392649203538895], [0.00693031121045351], [0.07298476994037628], [-0.014844009652733803], [0.013983488082885742], [0.0004492272564675659], [0.022267237305641174]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e8c68d651cc6360837e76864edda2792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-40.3016242980957], [-1.388820767402649], [0.27571114897727966], [-2.788365364074707], [-1.20401930809021], [-0.6939953565597534], [-3.698253870010376], [102.08600616455078], [3.887320041656494]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c85dd77d694b1d5e0857acc172bc356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[41.3016242980957], [2.3888206481933594], [0.724288821220398], [3.788365364074707], [2.20401930809021], [1.6939953565597534], [4.698253631591797], [-101.08600616455078], [-2.887320041656494]], dtype='float32').reshape([9, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21989ef841dbac669942fc7c29df1ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_abb85589745943fc4cdee303d090237f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3393935f880a5162958eb62b07cc2842
    def get_inputs(self):
        return [
            paddle.to_tensor(11673.541015625, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3f8eed46f1f6eec86205985dcaf04588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(1061.2310791015625, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c90e773c456927142568b0d6d1c77f91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.029102390632033348], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c16ff9b6503bf9abad6539aee42fc7fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.4080766439437866]], [[1.0083612203598022]], [[1.049621343612671]], [[1.0553209781646729]], [[1.6198089122772217]], [[1.0849533081054688]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9541c0d976646ae499f64bf41705104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[1.436981201171875]], [[1.051492691040039]], [[1.3073549270629883]], [[1.3907482624053955]], [[1.4530985355377197]], [[1.1249167919158936]]], dtype='float32').reshape([6, 1, 1]),
        ]



class PrimitiveOp_cef684ec1fb1449eb19dc278be4751b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 128.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_665d33dce9e8fc4bdd8faf761409cdae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cef684ec1fb1449eb19dc278be4751b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_57ea072be1db314765d7817c6fc73f20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_665d33dce9e8fc4bdd8faf761409cdae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cef684ec1fb1449eb19dc278be4751b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9345fd5ee71f8e373d018d000cf814ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50cd0c9f128018e8b291f53e2cc20011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50cd0c9f128018e8b291f53e2cc20011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_87bd4777e57a7d7b29f0453e85f4bd91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e82283802c03e01bb2b382eb8d072f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([5480, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b655a9449db312e2d5df8dad8cf8d330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_617c8adc197537c2b4e0acc696ac0bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb1cc3afa796f7d90832e82f6e6a58b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eb1cc3afa796f7d90832e82f6e6a58b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c3eac60b5a9cb230b3a34501dcf8e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fffd59306ccc2d69a209e3a5ffd9054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([86, 1000], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_702472987dc7d94212f5422c201c5d0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([54, 1000], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9379c0259067e71fb47bb60084697e22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.8375, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_542d7c4971f737ae9fbd6d1a5af1334d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9379c0259067e71fb47bb60084697e22
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4d23bd3db3fb3aac9596993cc6471c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f29ce3110ea8d2ef79166ca74b3fe7b
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c45d0e57de87145d0dadd9fe5052049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(89.05722045898438, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9ff87c32fab6c6f85580759417cb4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.33624267578125, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aaafcef19a8869b19e67f0510ad68a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4596b46ba897294bf283bdd5ff9a513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0ee6be1e392e9a3e5d62b265f7c7ad13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(6.2698893547058105, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_592ff4a1a7c890112fca9dbbcbb5e4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a2a84958f1079a014ecf754b23fb0453(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6697f02fb6382f58c9e6b30d0ff49dc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3087916374206543], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdb9b0dd4fb94c295d9da42bcc0632be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2936667799949646], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5c0533446fb39c546e09df57e5710fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.8984315395355225], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec2066eda573d0af5a97bad3b2ece9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f83777f27331e8106baf3bf44b1f6a7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_da632f6c8b7e41aec56de7cd63eac16c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_604d3e059d7d3b6ee2e24da9fd2ff72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(108.77389526367188, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c4cf957b44d7fa15b9ead63d26adac84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(62.76165008544922, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cef7152aba4331c91aa04fa47fcf1b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d3495fb43ea9d40660329e055376762a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_612c23b2bdbfd224d9b0af3a8fc80846
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_908a6a3d3c1505d8bc8b2fce3c348c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3636210858821869, 0.860653281211853, 0.45989561080932617, 0.293732613325119, 0.43766167759895325, 0.5128899812698364], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_766d0fa74fc5f089bcb5154172af5880(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.7485138177871704, 0.23536936938762665, 0.8053454160690308, 0.817694365978241, 0.6373422145843506, 0.4877123534679413], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c86d4522198d1336a1a99ac7ca9c2f64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4802755117416382, 0.17946738004684448, 0.5224344730377197, 0.0764428973197937, 0.7114992737770081, 0.6274797916412354], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c04301231200b660557fc2b683c6b415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5152817964553833, 0.3285520672798157, 0.5713917016983032, 0.4692542552947998, 0.36130884289741516, 0.497228741645813], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9de6f2ac338980c248c1d3386a099cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03883463516831398, 0.00019726960454136133, 0.024178875610232353, 0.002077957382425666, 0.0006825195741839707, 0.02110004797577858], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd41ad5d746fe9dcfa41f5fb49b2d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([0.017001358792185783, 0.11817431449890137, 0.014661362394690514, 0.04215633124113083, 0.03779536485671997, 0.003305346705019474], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7613c035f0612eff05bd3599e31bceda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09524781256914139, 0.1662207990884781, 0.04816321283578873, 0.137531116604805, 0.04644821956753731, 0.1607235223054886], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7893233c87139d0e8399aed82fa933f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b3f7cc21e270659f61f26eb35b4a9aa
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.232835590839386, 2.3465805053710938, 2.266756534576416, 0.7959005236625671, 1.4818192720413208, 2.3078603744506836], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_981fcbb7b7ed0d884998b2e0e8d75992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c2a25318c8711b8df066f6d92e085f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.015826720744371414, 0.0, 0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9030161ff106d4b1c6f5d1c10481d2bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_906f3f814a87b066634781484c02ba47
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0061447620391846, 3.231677532196045, 3.0824294090270996, 1.2567309141159058, 1.8899199962615967, 3.1586368083953857], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_24c16d46633730af30f324c69a86004a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([1.163149118423462, 3.2520620822906494, 2.711258888244629, 1.3589682579040527, 2.232752561569214, 2.4957945346832275], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6f227d535e1c605e9c5a9d7eb353302(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7c5427caa9c558a99961c0f0e44dcb9
    def get_inputs(self):
        return [
            paddle.to_tensor(2.202331066131592, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6cf706273c519442a8e69524121bdb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72ad25814effd01671cb784cf15c7917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_72ad25814effd01671cb784cf15c7917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_752f8ca56d98913bb94e91fabb0698cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8b61c939661591ef6d69711db474d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([1742, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a9cda92f3a94022cd3a0f2157e842e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40fa75bf5d0d8bce9d0b54e639ced419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_986a751708804f6977089e46628756fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 2, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e705d6ed092dbdba27a058e38e14e8ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986a751708804f6977089e46628756fd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e705d6ed092dbdba27a058e38e14e8ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986a751708804f6977089e46628756fd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a2987566b265afd2a51a990b7cd76e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aacc7050fa505f7ae6d99849a1bcbe9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_879f4f068236638f48f0d3f052f5277d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a2b5b84b4878eb3119203ab8c72257b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_879f4f068236638f48f0d3f052f5277d
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4aeaabf2ffe5491603361cd053f352b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a25879d80e0d7a369859a08c298bbfa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4aeaabf2ffe5491603361cd053f352b3
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6707fafd7597f1a40fdb44d6153d6dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8af85bf8446243c270dead072fecf1c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.1563340425491333]], [[0.04343500733375549]], [[0.008920107036828995]], [[0.14171743392944336]], [[0.42943236231803894]], [[0.43999648094177246]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_086c78e7c6498135923d8fe3b960887f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8af85bf8446243c270dead072fecf1c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.15264664590358734]], [[0.34894150495529175]], [[0.4814743995666504]], [[0.1594444066286087]], [[0.09437808394432068]], [[0.06603606790304184]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9c3643718de877e14d5b3773c9e7472a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9cf63cb73d4b66f144bde656f22adaf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.12701515853405]], [[0.04410155490040779]], [[0.37185001373291016]], [[0.14651024341583252]], [[0.4740857779979706]], [[0.1435951292514801]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9f1e1cf4f50aec51ce5dc3299ce6514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9cf63cb73d4b66f144bde656f22adaf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.46422049403190613]], [[0.49471724033355713]], [[0.21156202256679535]], [[0.0383564792573452]], [[0.4534785747528076]], [[0.36828336119651794]]], dtype='float32').reshape([6, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d7e2742e1387d50bc454aa464469833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d11b7df8be9d355244188c8ce55ee225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dcaa61bc25be87694eeb8e0803f86b7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23327752947807312], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7864d8d6bbe45af7738beefd542c9dda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ae0d8f715429fca2ca4937873768cd6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07856619358062744], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_10183bdef42dae8e340fd2fe148fea66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d843957af287465e1f7b6c54ebc20062
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11238446831703186], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b220f43f7336fbb582e0d220c95eb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d34303ca897f28117f320fb253b7a8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bae260dc4e989d3ea17690e4a60de19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c11ff0c29dd14d76cce66a10bb05cf29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2d68fbee2a467962b6c82eec367bb5e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_69ad9960cbb83e9ba60248b182dfae93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f3e8caf21e2b924e15a9fe21670a1c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c87fb3e4053e2331f92c24b0e1e7824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a73a7394824d9f993d437a4a45904e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bae260dc4e989d3ea17690e4a60de19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2cd3480cae684a818b4cae3d1973e286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2d68fbee2a467962b6c82eec367bb5e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21d8920587ca62ee19d883eabeff671d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4f01bc9d29d41fe65e4e254541a5473a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04613ce5044ff83df77851c8bc2624c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f0719bbc8de2496213e3d45e1a9e377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1639254093170166, 2.2044615745544434, 2.053987741470337, 2.108333110809326, 1.9586763381958008, 2.0109431743621826, 2.1483845710754395, 1.9205474853515625, 2.0875871181488037, 2.137115478515625, 1.9240128993988037, 2.0164082050323486, 2.0322937965393066, 1.9451545476913452, 2.0015172958374023, 2.25301456451416, 2.1551806926727295, 2.0124425888061523, 2.187988519668579, 2.0474181175231934, 2.0394086837768555, 2.271547317504883, 2.2335708141326904, 2.0531158447265625], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8235f7b2c6911ccfe70e41328ff8a510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(2.875631332397461, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f4bb0bf63c088bc9e134ea0147529a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2793837785720825], dtype='float32').reshape([1]),
        ]



class PrimitiveOp_83db89d9f12a246f47682b48d6e9fbb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec216e24e64f1df917bbda1644a7b8da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83db89d9f12a246f47682b48d6e9fbb4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.250755536471709], dtype='float64').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4fe9ecfa90d725c55a7b0728a3e8a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3b259419b07c3cca3c69649b6061c19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cc327cf986479818994fdc1d913124db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd012b37a24e162e8808b0c06f7efff2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_76c43c825cc9bb5a214eee54db5dc08d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2ad2316a9aaf8a25cbde6a9ce58899e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e15a3770c6e346a94c54fa040370b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e15a3770c6e346a94c54fa040370b1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_40034bd0ae0845dbf5bd27068b17217b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1423752e0e1991bf7e116c77ec7ed5f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([1527, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f1a6a7f274d73237b6fe94ac3f5150d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_640377f0bcd68149a7492333cba205ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecae9b65e6fc90d9f1642fc23446e14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ecae9b65e6fc90d9f1642fc23446e14a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_60d3513e6ff5a14f66b676a71181df9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf63a070c888da9eea92394a070557ba
    def get_inputs(self):
        return [
            paddle.to_tensor([[9]], dtype='int64').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_afcc92f6ea4bd1e0648c3983eaac25c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.2430911511182785], [0.23496943712234497]]], dtype='float32').reshape([1, 2, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1582c729164d300befc0f03f18256d12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e4728992888c5fbfc9aab6d0093d2260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c8d377f245868f63456a833c5d1abd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_574f28f3a6e812f59a875f18f4d2d8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5f6296811f6108c3c187c3b10459e4fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.1841213703155518, 1.872342824935913, 2.092287063598633, 2.162114143371582], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_798ad2d79dadd5c5125a999c9e4afb37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(0.17350471019744873, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f946b3a9fd9f7f31b5a44b7e48771f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_027e89de75ba9dd57da01cc23bfa7fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ef3c25b627b9ae981457d0a950967bf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e2c178f0086b85bdee37dbe284e6c183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(201.42575073242188, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f73ad2c54512b21597eeb1288d43051a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(5.126999855041504, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ddcf873901ad21706700249bfa4fb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(130.0355224609375, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a7eab85d032ce4bc6a7ab91c1bdd31c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.126100540161133, dtype='float32').reshape([]),
        ]



class PrimitiveOp_26ef4ea9e62df42cacf25e639e714a4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.25
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6e4341293a317fae29570e87cff3355d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26ef4ea9e62df42cacf25e639e714a4d
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a2fc90d0d6863de577e46efd1e1f118f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71b4204aa004edf2e935269fd9c98ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(163.173583984375, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebd84acaac05d2f90fd149b1d76d8bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(9.439167022705078, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0eff9b06d1799db9bd5ddcf25463c99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07083103060722351]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8982d7deb485de824b04c92d7dac5abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.012383874505758286]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ca811d25c49c9ec19afa4c39ba8585d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-6.719618320465088]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2988a239019edf7db297a4f29375be11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[7.719618320465088]], dtype='float32').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_92d77a627885a2ad898a525e56f9a393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0467146635055542], [-0.060889631509780884], [-0.02047661691904068], [-0.05326319485902786], [9.980075992643833e-05], [-0.037149399518966675]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21ab386e14a8b052d74ea27d170b24b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02797812409698963], [0.019563326612114906], [0.0015397187089547515], [-0.07706371694803238], [0.01333153247833252], [0.021454868838191032]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bc8f0a9308dc1518a5353fe5e06d002c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-2.6696853637695312], [-4.1124372482299805], [-14.298932075500488], [-0.30884212255477905], [-0.9925139546394348], [-2.731513738632202]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad00736971b7dfc0c3e82d2131814e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[3.6696853637695312], [5.1124372482299805], [15.298932075500488], [1.3088421821594238], [1.99251389503479], [3.731513738632202]], dtype='float32').reshape([6, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dc2507d5570d0c093e01b0b6d5590a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7782fedb136aa160b554a62eba14c90d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.1637033373117447]]], [[[0.5363078713417053]]], [[[0.7413976788520813]]], [[[0.9479138255119324]]], [[[0.3223705291748047]]], [[[0.9293025135993958]]], [[[0.3269089162349701]]], [[[0.5215974450111389]]], [[[0.9874126315116882]]], [[[0.7935583591461182]]], [[[0.5209171175956726]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b49a170875cebe9c908dc33771df5957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41cc2c63c8cec85a926ab2a4e2ebbd8c
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e01294341a9bfa393f00222585ff91ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype='float32').reshape([14]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a96098767cf34d7491f939b8cfe2e2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5], dtype='float32').reshape([14]),
        ]



class PrimitiveOp_0752841595e2c6f256f46fb640dc2989(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b1e7446557ec757e60df5114a560a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0752841595e2c6f256f46fb640dc2989
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b1e7446557ec757e60df5114a560a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0752841595e2c6f256f46fb640dc2989
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_baf13dd4dcf82e4350f7bb1045b15c5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 80, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df1f8d042a4757e21321b9e89314e2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baf13dd4dcf82e4350f7bb1045b15c5d
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df1f8d042a4757e21321b9e89314e2ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_baf13dd4dcf82e4350f7bb1045b15c5d
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9470f10a6fea3cf37e787d87f2034760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0], dtype='float32').reshape([28]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63d3aab40e19096d69b82210dc3d4b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5], dtype='float32').reshape([28]),
        ]



class PrimitiveOp_3ac7a179da2ed1a7c72bbcbaf39200a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5d7929d32bcc59160d79d0f7daee694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac7a179da2ed1a7c72bbcbaf39200a6
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5d7929d32bcc59160d79d0f7daee694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac7a179da2ed1a7c72bbcbaf39200a6
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_12278475bc7647766ccea164ee8c32b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 40, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c06a5d2eeb0d88f7848113b8c9a55a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12278475bc7647766ccea164ee8c32b4
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c06a5d2eeb0d88f7848113b8c9a55a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12278475bc7647766ccea164ee8c32b4
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_245d904838b4d42cdf1e0e327e8e894e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_896f484ff41551dd6c2471b11eac6da4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e29daf00a65f612f8bd87cdfefba2fad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8feecb99d49b9560487aa0c657475427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e29daf00a65f612f8bd87cdfefba2fad
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8feecb99d49b9560487aa0c657475427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e29daf00a65f612f8bd87cdfefba2fad
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_cef0463eb94255bb2f1fbddc06b8372a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 20, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_749d2fcfbc1ec3fc5b078234bbc293eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cef0463eb94255bb2f1fbddc06b8372a
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_749d2fcfbc1ec3fc5b078234bbc293eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cef0463eb94255bb2f1fbddc06b8372a
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19389b2b3dcbeffd7ad11b742e8f54ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_80eaefb5be14732f7fdab0a8d543f81c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(16, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6081e0fedd7d2290fb7e174e765884e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6081e0fedd7d2290fb7e174e765884e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c2836f20fd9d1f57b1171292c96593b
    def get_inputs(self):
        return [
            paddle.to_tensor(13, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9e489f29600da1609784b7a5c6ef7a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.to_tensor([[3]], dtype='int64').reshape([1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_67e767ab5cea20c26edd0b61d068d5e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58bc8f5029d45cf1c4fdc3e7b4f0ed04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.24354983866214752]]], dtype='float32').reshape([1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a06c9b456a560bc790970c1306275e3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46ae38bd009df1f60144532fa5485775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017508ecc884b1da4805ad4be833af2d
    def get_inputs(self):
        return [
            paddle.to_tensor(4.0, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7d0a2499e0c3a731bdfec9e15895eef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_017508ecc884b1da4805ad4be833af2d
    def get_inputs(self):
        return [
            paddle.to_tensor(7.0, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f4d1bcb1b1017b94bce1cd3fc42d6bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59ddaf42fe225ca800031751ccb16c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(62.63297653198242, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_311f6d1f62e5ba0078e4e95adc0b16ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(2.0307183265686035, dtype='float32').reshape([]),
        ]



class PrimitiveOp_2bb6b109a4389de1c28a1dbdb02b6041(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.1, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_162b3e06ea80e21e1c655673e6b9330f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bb6b109a4389de1c28a1dbdb02b6041
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddd5e30a73a679891d48f2c1e5ccee00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47f3cf19ab64b58946acaea7546880e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a0dca3c03fc74062cd2e7c65340e3db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a0d8e0cefda1690ec3714024c9aefcd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f8b561adc88fd6f915d5e5acc529efff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2fbfbce2bd7381e95c3d53122e5498fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9858c6a4010df30f8773c094f427e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(156.8470001220703, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd568336ff8e7383b457fbf08b3ca2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(65.62250518798828, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95d22ff1f98388dba594f407cb28d8f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9cf105de78510ce1ed03a167bd72482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a4c6a47120cafe32ca89d6e4e6ba1a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e36340c58551b822ab72ded8135d8b9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e36340c58551b822ab72ded8135d8b9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7178a062193b12716b5e805786d1f47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a1d1869e5109d910e5ff9941680dc5ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([2066, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f9902106d7b6fce9ac0c0ba628676c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5776d64a833c62d5c9f202ba73170e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c2cebc18c57626331650f3aa797b08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c2cebc18c57626331650f3aa797b08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58381133f4539e4bf7fc47cb33959aa7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bb6b109a4389de1c28a1dbdb02b6041
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_251b02856772b603f655e55bf53d5254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(151.71063232421875, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54b19aa105571f6593c99457ac76ceb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.160825729370117, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28cff3e60a2d5a8e6b1303f77cff3710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a3e1edfd7aa2d8449adf2abb79dabbbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5], dtype='float32').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c113753be073667135ff4459e0bfd7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01693892292678356], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a0b11311b3400a5c069ec5907e7da52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1082133799791336], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d45c0af68dd3d9abb7fd3426fd6be5f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2033243030309677], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_01e7a1c109985a61b81c5eaf9c50d775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2884811758995056], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46f44e026b423e415b406fec497edd4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1573590785264969], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b45c89f7a9a383ebdd66f6d2b1e22f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5564619302749634], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6391b351c207b234d90d62a2afcbb12c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1313057392835617], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3253ba47a206194e32dff365d02c5b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30344873666763306], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efb3a83b86655b2924f74e6b110b81bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12425953149795532], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f339f768bd065864ff06791b9a8bd09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39257174730300903], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fbbf98948507debfd0a909588e8979ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18483184278011322], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c60c78f9169665d13526835593e6d17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33507657051086426], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf5672d4696fa8d7c04b5d1e8995f3ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2986142933368683], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_68e573859d11ced05dd003407d37707c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33657583594322205], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae835c81b6fe6a869f6dd3906017c8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.009820956736803055], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e81b0f59e1e4581201348eb2d0bc8f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.061011020094156265], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0de54c6114281e659fd2eed887789f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2a84958f1079a014ecf754b23fb0453
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17482644319534302], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6316c2242466ffc7e6ab3f513d82bd9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45183873176574707], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ea0daf5a3d66e6022b2388d0789519ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09283372014760971], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e20514c36ef6090f564b38a33e6bdddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10435719788074493], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2751188c38d47c65532bbefca0da492a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4671162962913513], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ad2150e0f25805b684d2fb51ea6bf104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4765782356262207], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7cbb2f2c404ea4d1dead3b01a1edac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7fe9c8f2c4d8857f68188775e865b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7bf06456ba5f1b21eec14a44f51bd9a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aaeee1ea2f00d688a8acdbd928083be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e9616412f30553617052bea660ea0f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41582db8f3d96b3a57cf420473e02ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_1d92764b51dcc481e68f7f2d572cb9c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 16.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7f9fb0e312c9cb0f0b70f81ed33182f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d92764b51dcc481e68f7f2d572cb9c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ead80f73ea9bb55f713a5eaf2ed43776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f7f9fb0e312c9cb0f0b70f81ed33182f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d92764b51dcc481e68f7f2d572cb9c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91ebba41f91179a48e598da2678b1b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_91ebba41f91179a48e598da2678b1b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_debd201c300d7b31f3df46f25605f28c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_12f156c9f3cd127dadf27eab3e7fb9b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([4586, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae2f73549c3c5ee4ce8be3acd258a2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04803b230444875b491e968475475313(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26f836038c632f660b1199394cbaf871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26f836038c632f660b1199394cbaf871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b1c750136271e02015adf99b0be3a21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6223eb60e7c2a05de6a6d7544abb59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(98.3559341430664, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b053c75843de399c098ccf1c4d004cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(300.5543212890625, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89ca43fc95332731c0894fb0bfb68019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_89ca43fc95332731c0894fb0bfb68019(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ddfd7247b5d1c739783698a8aed1c545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b499c8b0e6f74b7a6fbb37f2b7f161b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([1073, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f50155ded684a2497ae3dfe67d51f61a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07759ea819fd36f4737872e629444c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2dad212499cf84673edf7f0c46128ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2dad212499cf84673edf7f0c46128ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f281d1ef6e1958938d25e3cb5021c1e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74ff518ece4646109f1a4ba1bb895248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58c3dfafa5346b2102ef48bbc3dd2d44
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ae6026644ca29cf5aebd9e38f4b8aed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cfda4c2d4cac996d001acbe86d75539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(94.67943572998047, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3e4996278c648f3d447503e6ea19d98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(3.4263734817504883, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7184b8d8706c603f49e497b3aaf9d397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed0049f1eb88a089f99b87a741b696bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa3109363ce374c82889b389a3686dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5e2da4bd16ae078d6c2ceb27af813ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ec14cf2aff99c930db1fc9ec5cda769f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73cf776fd1e94655b46c3377445da2d3
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37f26ed530297ac6971a933712f7e6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07b61a8de40c57ebc7e70d65060b8961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd684732a3c7cdc147490e2264701c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2c7504e0f7da0795de0d419c1c675a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ed0646c0443c50b6c8ea44ba4b70ee
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07b61a8de40c57ebc7e70d65060b8961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cd684732a3c7cdc147490e2264701c0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9604c8ff95876414b086347ce35e0e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_66ca5a8b0cffd6b5992d71a3e7804c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5c083fc411641973e690bb7c61cd0e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9bdd2ff2c81bb6ad57899b9fbfa0fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_36c2e887895b8173fa1d46d3fd2cdaf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c02727c2ab22910445ddb874d528ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_baa95c755a1943e98a13c19ea3355e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_34a75f05e8fe0f24402bb92d58589cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b09f5a00387eaf0e2304eedd9dbd0595(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_828062425350c4087b2ad1c778d4ddc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0062c005ec55be72835accbda4d0a39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_36c2e887895b8173fa1d46d3fd2cdaf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9876a18a4115ff6043d72f1990cb048d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_baa95c755a1943e98a13c19ea3355e9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0d49cdb9b6c3426b8718f6ea60c9e5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype='float32').reshape([17]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4438e3d08730ca1850edb22d24e2fe80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], dtype='float32').reshape([17]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c3eac60b5a9cb230b3a34501dcf8e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c3eac60b5a9cb230b3a34501dcf8e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c3eac60b5a9cb230b3a34501dcf8e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7075034132b6de6923969bd4d4e142c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_272942409ea561f3f00f4e7afffb6d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73cf776fd1e94655b46c3377445da2d3
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d5c6237b36e9ba428c543c50c02d49fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538025fe57de5047eb05d808f78021d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52400e5229c03e51d71877a026e857d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf7d0c12bb9357fedb433b365f5956bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6ed0646c0443c50b6c8ea44ba4b70ee
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538025fe57de5047eb05d808f78021d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_622e085199e74a32651b79ac5ef475e7
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52400e5229c03e51d71877a026e857d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba998a3710cdce2acbd257caf0d9d41
    def get_inputs(self):
        return [
            paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17c1177690e6263ad4e584dca8710357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.07234402745962143], [-0.048308003693819046], [-0.026102447882294655], [-0.060440488159656525], [0.03520594909787178]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5cf9608c70b472788dfe5ce3a0a5d070(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.025636205449700356], [0.07950747758150101], [0.0040383669547736645], [0.08718772977590561], [0.03009037673473358]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4541d507325f836d5fd92efa01b204a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.821947693824768], [-1.6075907945632935], [-7.4636149406433105], [-1.6932224035263062], [0.1700069159269333]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9dee8d39fd9eadcdcf9accc51e0f89da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.8219476938247681], [2.607590675354004], [8.463615417480469], [2.6932225227355957], [0.8299930691719055]], dtype='float32').reshape([5, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4dae6a5fc2da5a41131f393f67c9e1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b224087c01ad6d103c153c2f7172338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93e5217e49ef45f509acff8a6e0fe6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf2849a29c533bbc9eab239a6da17c19
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b224087c01ad6d103c153c2f7172338e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_93e5217e49ef45f509acff8a6e0fe6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf2849a29c533bbc9eab239a6da17c19
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], dtype='float32').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e5f31a2f88c39611800cf5b3e6ba1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09308fc862b2a19261dec7942f8f0f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec37bee4e48a95c86d20024ad5eaf317
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e5f31a2f88c39611800cf5b3e6ba1bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32').reshape([8]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09308fc862b2a19261dec7942f8f0f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec37bee4e48a95c86d20024ad5eaf317
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], dtype='float32').reshape([8]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd9c68a5139b98bb97c1ec0a6819a279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0fd54ce7bea39bdc147e5f99394d351c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95f93f56a3153ec2ae6d2f0dd113d428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95f93f56a3153ec2ae6d2f0dd113d428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95f93f56a3153ec2ae6d2f0dd113d428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_be1dcee5c8246d39f5d03b29d6006265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_7df6a4f3356b708be6faf4d09aa65d09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 8.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b474a624dae3ee689eaee464a35c403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df6a4f3356b708be6faf4d09aa65d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50d3b95019431904c045b82b99761da2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b474a624dae3ee689eaee464a35c403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df6a4f3356b708be6faf4d09aa65d09
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c3d249d43fe80355bed1630c616d284b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2480415f0c799c9bf7ab726729d6750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5ebc127f2cc30fe443dff87b2144036a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e5c222b6a187ee0a0a552b280d74163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58dcaf5d14ec113856aa372c89bf1b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_849157daa5639426c4c5a3dba4dd79dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b2f193c3a6ede418d4aa277c5bef4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2853190004825592], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8852e9c12ee80601291401dab1a27d48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ae0d8f715429fca2ca4937873768cd6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19351647794246674], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7f2c079d3f264d1b03949eab68825635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1f5ad7cb18129158ed60600ecae3787
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04894401505589485], dtype='float32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cea2ae7233604f93fa5fa5c80fc2a909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_367d1f6aec56641187e36666d5804fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aa7b7d93b00ca94675f8db2b6b1bd2a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa877fdc839c3d9af416ed8f2baeffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ed850dc359f3e7a8a2e4fbbf482cda3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa3fa91417b8cdf100453dcd4312460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa3fa91417b8cdf100453dcd4312460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c2d3cd7caac8f444a27fb2719e1d16a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de2f3ff2e82a18b776edb980beb0c6af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([2383, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0b622cbe9a1729149d7fe40b0634edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_136935abd49c5f1f5996d65b9fe44e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c797d3134d0a027d4d7646b24d782f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c797d3134d0a027d4d7646b24d782f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_22723d2a9477329f5ab9c1862ed9424d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_22723d2a9477329f5ab9c1862ed9424d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2de1bec684a0019ddb201089171f7429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8dbb0b200bff210388e5fc33ecba181b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([3030, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5e0ea9a6ffb0c017145f58a6bd0ef534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_61ffb7f90a8b2f1984332e1389054739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db7093e32340686a877dccecf87247c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db7093e32340686a877dccecf87247c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d54a060680d92fcfb6901d6eb4ccfdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6d54a060680d92fcfb6901d6eb4ccfdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c49179f09fa8ccbc41dbb01516b6b180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49a9c43f3c5da5efa4cec93504d7a9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([3787, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_53ad487cfa8c3a9a22a693ae7ca9acd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1f7a786b6047508c56f1fcffe339bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_be1c242581e9d15f9e8a4a0fdd9edead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_be1c242581e9d15f9e8a4a0fdd9edead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]



class PrimitiveOp_c2aeab007b853f88a879350bbc3c9838(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 64.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a4350cd4011505701213029ea6bc599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2aeab007b853f88a879350bbc3c9838
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b975ccdc0a9307389b6a396c62393b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a4350cd4011505701213029ea6bc599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2aeab007b853f88a879350bbc3c9838
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9de89a4432864df9169936136d168384(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a9f4e9c3675a8d81843639cb50f2bab
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9ffa76f0ed5efef35cdbda0d778f736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1391b3865563dd0fe8b90f752ee4412
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_233cf009bdf5070e5a30f77f142512f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5134b984a39062791624c7d43726b62d
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3278a2169a77d803b26189b6a903353d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 0.925, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c49578f7468c0491c7f3e3d9a18a197e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3278a2169a77d803b26189b6a903353d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5860388875007629]]], [[[0.8120017051696777]]], [[[0.9539660811424255]]], [[[0.8134338855743408]]], [[[0.6751497983932495]]], [[[0.6726080179214478]]], [[[0.27094537019729614]]], [[[0.2922995686531067]]], [[[0.2422485202550888]]], [[[0.689205527305603]]], [[[0.37748992443084717]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ae556f75ee7e2227d7c181f4fe5c940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_443507fdb20b14fefeed764e57c7e047
    def get_inputs(self):
        return [
            paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e7cbb2f2c404ea4d1dead3b01a1edac8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_956719df85894c04c8e216c6a25355cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0c74c7b7002ee25eb4784134c3583c2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_48a971fd3173758feafffb2bf860bfe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c8ac4536e4ff567573fa673c8bae5897(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f9617da81e7c2d16c88662cdd48e773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8494bd16ccd08d26ac6970ce919fb9e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ca1d8faf7f450ecc47775c2bcb90f4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2844faf1e5c0e0174f209066220d23af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b392a4ddf0f96225c3861ea6b3f823ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d3e6912cec8d02b8c60325cadb8ff64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b392a4ddf0f96225c3861ea6b3f823ee
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8d3e6912cec8d02b8c60325cadb8ff64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b392a4ddf0f96225c3861ea6b3f823ee
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_96457a37be1890667dc238e2d338e9da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 32, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05ac626fbcf4fd4dd4772906220ad653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96457a37be1890667dc238e2d338e9da
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05ac626fbcf4fd4dd4772906220ad653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96457a37be1890667dc238e2d338e9da
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6006bd915d6f7685d4c2e9c12c23eb99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_36b1b347524c9d5e685a1a1ae067f3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1cb1ac6f613e78a60b0595492e1b15a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1dbb61777e7ea63610e68191cb45c932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b634e5353aec49abf03af4c1e8f28d6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6e602e49dcf3f5ace3139c0b01d63f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b634e5353aec49abf03af4c1e8f28d6f
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6e602e49dcf3f5ace3139c0b01d63f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b634e5353aec49abf03af4c1e8f28d6f
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_079f328bc5c078bb71fa7022674ec668(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 64, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f98ed5c742b6ada7aeed1899e66263b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079f328bc5c078bb71fa7022674ec668
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f98ed5c742b6ada7aeed1899e66263b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_079f328bc5c078bb71fa7022674ec668
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2837cc2afe16f0a6622585a4e34fe81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5c9030fa0acfcb41dcc2aa9f7039aa89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a5bfa8173cf0c06f2593980605403ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype='float32').reshape([25]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e7491c5fa931b1186c4ea3f264be4de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], dtype='float32').reshape([25]),
        ]



class PrimitiveOp_9759ac610c591e45759ae01be7fc1b21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aed1abd1f91f2edbe8be80901ecf9c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9759ac610c591e45759ae01be7fc1b21
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aed1abd1f91f2edbe8be80901ecf9c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9759ac610c591e45759ae01be7fc1b21
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2dc59a0c171bbb79c45275373a353431(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 128, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_85e3ca963510a4d49cd2a03deeb6206f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc59a0c171bbb79c45275373a353431
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_85e3ca963510a4d49cd2a03deeb6206f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc59a0c171bbb79c45275373a353431
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5f93c7797df19fb752e06eb46a5d673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], dtype='float32').reshape([19]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_727d47a2b53251e61aaf214a6b6cdbe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf2849a29c533bbc9eab239a6da17c19
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5], dtype='float32').reshape([19]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9e56a9506f5682af9656dfdbb82a3df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype='float32').reshape([13]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_12d4172c7430d5cfa5028d4052a27b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf2849a29c533bbc9eab239a6da17c19
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype='float32').reshape([13]),
        ]



class PrimitiveOp_73e62474c4ab21939512956414b709d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79dbe1ad92807776c11ec190d337dd94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73e62474c4ab21939512956414b709d3
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79dbe1ad92807776c11ec190d337dd94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73e62474c4ab21939512956414b709d3
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_05edc2f9b29607fb38bf866c93408231(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 256, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed7ae9f5517735ae3af145245e6b0a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05edc2f9b29607fb38bf866c93408231
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed7ae9f5517735ae3af145245e6b0a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05edc2f9b29607fb38bf866c93408231
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f26955f50013363d6bf490e5cd00466b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype='float32').reshape([10]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a4d5dd1010caa7b1331107fd5618013c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec37bee4e48a95c86d20024ad5eaf317
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], dtype='float32').reshape([10]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b5b80fd21f12bbd9eb8d6da0955961f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype='float32').reshape([7]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a1bd2117e3166846c16137704bca5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec37bee4e48a95c86d20024ad5eaf317
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype='float32').reshape([7]),
        ]



class PrimitiveOp_9b9041fa3b21cc5bae168c9b32634991(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, -512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b71eccf9726d7bfa27fbe8697bd28ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b9041fa3b21cc5bae168c9b32634991
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b71eccf9726d7bfa27fbe8697bd28ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b9041fa3b21cc5bae168c9b32634991
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e1916ea11c5781b67d196ad8692dcf66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        return paddle._C_ops.scale(input_0, input_1, 512, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_944e32870310f93d37aaecbdbc364b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1916ea11c5781b67d196ad8692dcf66
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_944e32870310f93d37aaecbdbc364b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1916ea11c5781b67d196ad8692dcf66
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f6e72cefdb7c530b8bc0f2bf2316b0a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_295d26bd8810000771e31c624e39f330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b7c5244e75aaffe719d961ffcd5d284f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2c21c1ecf5578f0b6b651f72f17b3e8
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a37c8121f1b45eec87d03f896d28dbd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c199b0f7d8d81702fc1ed5c2625bd780
    def get_inputs(self):
        return [
            paddle.to_tensor([2.116051197052002, 2.2622504234313965, 1.9536495208740234, 2.0681753158569336, 2.0806095600128174, 2.11643123626709, 2.0751419067382812, 2.1351099014282227, 2.1985647678375244, 2.044846534729004, 2.0799427032470703, 1.9587132930755615, 2.056234359741211, 1.8934751749038696, 2.1239142417907715, 1.9864888191223145, 2.1752965450286865, 1.9954736232757568, 2.161653518676758, 2.157972812652588], dtype='float32').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_736c06a2bee6ce7510a1ba7747b15882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_739dc0984d342d6c707069659dfcc435
    def get_inputs(self):
        return [
            paddle.to_tensor(2.5518760681152344, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d4e858d4a2933442a8fc65cb5ce369a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bd5cd710c83f3628303942bb70517f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_17540281007f73c6237047303b960486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(366.6561584472656, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b5ad5c2a74007d9ee810f49933cdf092(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b8f45839b32845f8f66671f8e3f865bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.006892362609505653], [-0.06635904312133789], [0.016340315341949463], [0.024487247690558434]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a5b3ac1c1b89a7f723c2cc61a062ee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.022870756685733795], [0.04407133534550667], [0.00013807692448608577], [0.05499931797385216]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_99aaaa30c051658c056794531c7276a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.to_tensor([[-1.301361322402954], [-2.505718946456909], [117.34202575683594], [-0.5547717809677124]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7346d950bb89736c4ab0569783739468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87736f1f5873a74b917244bf139b916c
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.301361322402954], [3.505718946456909], [-116.34202575683594], [1.5547717809677124]], dtype='float32').reshape([4, 1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cf1dec75865eaf7a5b5f53d364f52fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_334576c0905dfb146a57147f41aa83fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(33.52735137939453, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4160c870713359a2eeedcbe59c279ace(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_962946b81397f706108d80fddc5f1950
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44230620cfbf1d420a1008f5e55b740e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bbc8818a02fa79df132b41b02f5557f
    def get_inputs(self):
        return [
            paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05536a99008b9c00f20fe883a4ccb2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05536a99008b9c00f20fe883a4ccb2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b295dedbf78769e649587f9fd1634ecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_924028a84909f656233ac5fb4bff7787(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([2084, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_633c24f855d65f85b1d20c23339c88b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdacb3b94f2a6d885f277aeb1d1d3450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f985bbcacd577deb0488abf91e23b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8f985bbcacd577deb0488abf91e23b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_752da18c211923d4e4b396a27ebad5b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e2b8e7ba257bd935df537f9c7c95d2b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_404c1a7726f8f45c98d9721630790164(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a614c0d6329c8cf3e6c3b95f1163f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42a8ac3599d288b24df0b8e0cd554574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e5980e8d3cba549743de3f17c3a4e799(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 32.0
        return paddle._C_ops.scale(input_0, input_1, 0, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_096347e09fb43054c15c67f816f56b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5980e8d3cba549743de3f17c3a4e799
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_90fbeed70cd6450c4591120e72c88ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ba9b11fea1e7751de69fe7517dd91cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_096347e09fb43054c15c67f816f56b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5980e8d3cba549743de3f17c3a4e799
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c8c8f358ea0403638fa36416d1f4d1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_692787bef9edd3d8d3323117c846490c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fad4f214604305e3f5baf9e73a5af9cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9598a42062c678694545bcaeb6663c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7df0f1fdc30631d8cc364216ddbb4b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9efb9e882b0bdecadacf79d501598aa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75f0dfc96865449d5e162de865151a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_26db7685ca0c0084e2b8c412fe0291ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5756099675c3d1c9ae3546bdc4ffc8bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dac75af7fd0e03ffe04021ea6de21a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_537b4d39eadbb0ac1bc7f5dac3b7a77e
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f314a63940f13e8f821846892a5dfced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7df0f1fdc30631d8cc364216ddbb4b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44420b9ccdb066de6c67c979e641fc2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_75f0dfc96865449d5e162de865151a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_087df866f7af213e860487050dc3fd65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0], dtype='float32').reshape([18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9367fd6c99603bdfaebf16ea928c28e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.to_tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], dtype='float32').reshape([18]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f38d285fc84096200097eb6bbef8c6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(92.70454406738281, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fef9726a7cf05058728d1e5735c69b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(1.8303749561309814, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_58dcaf5d14ec113856aa372c89bf1b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_590afcbd8d4ed4bf2211bc383f8fa29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a092c4eddcddf77e83403f110ce2b2c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c031ef7d20faa1623bc39f31139d008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7281245211bb88cce6849f1deec4da4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aee5d09102bb8b1d9fdb9607c7600f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_95f93f56a3153ec2ae6d2f0dd113d428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_346c04be3fb09e2ee5516f37e1989676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5692fa11fa2c48bf0ef3cfd8a158e0a2
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b220f43f7336fbb582e0d220c95eb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4fb4d2c77644a23116f1ab7295aeb5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39fca2bebd4243f221f8a3f0503b878d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c425c4fc58d705f167c9086bf7833522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c6662722abeaf266d14fd1d5a4d865c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_700fec5af778fb2dbab2aa06d008a408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(37.852294921875, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f7aff5db603a6c2e7fa0bac0d8cc94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f35501f608ef1621d0f640c56074e83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61df107e237a575a6a230d9e977d1186
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa11154391acd587d8f7b8f3e37e9f9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_aa11154391acd587d8f7b8f3e37e9f9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85adc26cce9d30fe69880ce7511d3dad
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4bf82365b0e474014a5ff624b16f406b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e657b42ecb4731190c10a4199dda8f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee5d735d29852902c293c308523e0806
    def get_inputs(self):
        return [
            paddle.uniform([4260, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c308d9e8ca258cce1ec0f5db483c52d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05bd670fe3209a01015ba98799305f3a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05c758ae9f569e8975236bd03d20ad81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_235493aee458c3fb166ca477f768df3d
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60e6155b9e792a3fe4636a361fa1a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f60e6155b9e792a3fe4636a361fa1a32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17fda862cefb7d6a648f2b119ec506ad
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fa2f57476009d9a37e34bc06e6c21140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b61d81fa8ffb0a248d32fa365cea1b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_526eacee885915fceaa55e37399e5a90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f36eae1028c702a1df0ddf15af28fae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(191.19805908203125, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52da9517138f39d5df9e475445482aa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b674146c71585e699d219730cca623b6
    def get_inputs(self):
        return [
            paddle.to_tensor(4.286322593688965, dtype='float32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c2d81a59bb3cf31ef021a5d29dd7cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d26381a4215b417dfbcd83b353f99510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3ef78629590f6107c4e398da8d0f493
    def get_inputs(self):
        return [
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5913f3b8efae456dff8ecd0202a0bd53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_964d1cbb3c19551fd93e87e5a42355dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea6f6ae5c03d88231ec8b8f364ec3f9f
    def get_inputs(self):
        return [
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffd5734a33e6a8d86fc194091c2c0d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dba1d1a557c3d9f120988710fbfb8d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2f5c9b01dab1c3755b7c1945012e5304(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d99b8cf51b8fa7a688e14379c7b62d6d
    def get_inputs(self):
        return [
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4878b67d8794f6d935f54402e568687d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11f94ac04f74375a9bceb5e475d7f032
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0f7aff5db603a6c2e7fa0bac0d8cc94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e85ff6d8683abec5e6effa69de371bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_383600e57e743bd3fffb65d12328aa24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3948a2e49bcfe0864ee7e556bdbd63b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_35a3763bba0380cbb1f9994b676df33c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740455d0ce9e07b21ec9aa923db6dab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ed006e1d24a94bc42198c176f109b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6d970b8e038568d92d479283daa5d3
    def get_inputs(self):
        return [
            paddle.to_tensor(36.31705856323242, dtype='float32').reshape([]),
        ]




if __name__ == '__main__':
    unittest.main()