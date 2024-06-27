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
    class PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a7cca2ebc4e917e2e81680f43ceddc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_478c7e17f9568a938ce46a7c8b0b83f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8d839cbdb2c01a35795e873827a6100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e76d6c8c9bf76e5c4afd4cfcd36a9be0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_462d17dc2d9ba74bac96ad52e605c162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1a7cca2ebc4e917e2e81680f43ceddc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6f4eb50a69ce7d2077189aec7e2b2190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_605285c7b3d6e85d4cd54e5144d0753e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_21ed57a976ba8ec90dc089f244e6ed11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_3a371634fcbc29254f27cfce25559772(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc6f86078fa5355e25063ad2e718e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_90b537f2b33d865928bb11aff3a69ac6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e81ccfdd81c5d397100d8746fad2942a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_91b13fbd4673ef723c37cb27075c39ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_91b13fbd4673ef723c37cb27075c39ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_dc6f86078fa5355e25063ad2e718e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2238871be7af953303cb82e6e1cbd908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2238871be7af953303cb82e6e1cbd908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4571dfaee0fea4bcd5e18b2f3cb68e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_100c045f7d05c65734cd7d2bb85f57e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_22eb895348efd19228ccb67d1c70f57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6162465b11661a89f607030f3c00a5f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_45948fdca2dc3b4bf5444313466a75a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9fc82a179a328912b06c8cac475c65b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d4188bf1878030ca5177e317dbeabc3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_134399d0256b527981bedbd745445f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2135d500adee5c985c0c282ebbf6d743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7271bcd12b0c26ca0c5f66dbb533bcc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f749b8cf6feb3dbfaec8e4f1246d5e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e979ca890e7f3ac15659de72040927e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e979ca890e7f3ac15659de72040927e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4571dfaee0fea4bcd5e18b2f3cb68e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2238871be7af953303cb82e6e1cbd908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2135d500adee5c985c0c282ebbf6d743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1a7cca2ebc4e917e2e81680f43ceddc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7271bcd12b0c26ca0c5f66dbb533bcc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9fc82a179a328912b06c8cac475c65b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_39dc98983b3b04ea258dd8fe0e5db3d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_19c6c4947b0112b39602c922933ac69a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4571dfaee0fea4bcd5e18b2f3cb68e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a371634fcbc29254f27cfce25559772
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_552a6d7ceb14455ee72b6b494fb4d219(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a28c4cf9b50742c7d78bbfab1c2efba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_552a6d7ceb14455ee72b6b494fb4d219
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_c6d09dd30ecd5ba40dce613f3c9d5c93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36d37fb50dcfb81a60329283b40535a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d09dd30ecd5ba40dce613f3c9d5c93
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_be8a989d6712b608028f85bca1bc9e3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86970], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e8c8ac4b52dd22549f2892009ca3bd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8a989d6712b608028f85bca1bc9e3e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_53f802977e391d1be6852a136e5b78b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[242991], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9c25eb35a468b9ab216ef1854fbccb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53f802977e391d1be6852a136e5b78b0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_bb772f9c0e269e24f06a78c709293771(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14b5eca847fa2fa0ee44602cd502a0ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb772f9c0e269e24f06a78c709293771
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5a28c4cf9b50742c7d78bbfab1c2efba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_552a6d7ceb14455ee72b6b494fb4d219
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_1ae286d205f80bd5382014ac0262b025(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[220968], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06ee291946901b0c29a9deaf3aba21e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ae286d205f80bd5382014ac0262b025
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_cf35dd61096528b782f4c43c83a06275(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[153450], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab45395f1182c698d9fe5d24c8bb4eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf35dd61096528b782f4c43c83a06275
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_d5c69f5004fef8fffa3b23c032c46643(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185691], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8eac0dd450f3e2db015f29ac17972db0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5c69f5004fef8fffa3b23c032c46643
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e50501c5ddf7ba5b0f7d79e91c660e55(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2bbf036f98a015ae6f4fc2b3fa024536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e50501c5ddf7ba5b0f7d79e91c660e55
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9bc26ff406a7a2fb459ed2a22d4cdc6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[113061], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e224d336eb0600889e6ab96501238beb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9bc26ff406a7a2fb459ed2a22d4cdc6a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_8068aa076a8f63200d75976a1d2c234f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_ca250d4ab2bceacfb296ec256f68a831(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46e229235532f09c43b732f386a5c5f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca250d4ab2bceacfb296ec256f68a831
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_43f03188d55e40bd73c480dcfe1b707c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_554c41bd1d2d0943b658457bc6822980(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43f03188d55e40bd73c480dcfe1b707c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_554c41bd1d2d0943b658457bc6822980(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43f03188d55e40bd73c480dcfe1b707c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2bbf036f98a015ae6f4fc2b3fa024536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e50501c5ddf7ba5b0f7d79e91c660e55
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_0b19a09d0f1c9b07b47cfbbeff5ba671(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25b15ae027d48e447546907f0a61b847(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b19a09d0f1c9b07b47cfbbeff5ba671
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_25b15ae027d48e447546907f0a61b847(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b19a09d0f1c9b07b47cfbbeff5ba671
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e491f245cb708b93db7b16c0ebde4582(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ac6b2391d15e4f2252cd76e92d8b5b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e491f245cb708b93db7b16c0ebde4582
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_708a4676aff1b49699d32cb3f2b5def4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[205923], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ff113d25eaea91fcebea99df40aa40c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_708a4676aff1b49699d32cb3f2b5def4
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_978fe47038634004150db57c27902c21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e70ad26955c46bde1c786499deba15b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_978fe47038634004150db57c27902c21
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_df957f8ec969856409fbce251a17da6c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[123783], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73f2666ade2ac0f63a473bbc33b990c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df957f8ec969856409fbce251a17da6c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_78324d803f3f1910c1fe16f6cc28102f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171888], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00bbf99533aa7014ca83c119d18d5111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78324d803f3f1910c1fe16f6cc28102f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_2f15b1d759ec8dc197ce8fb6976916b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ab586694a24fc6cc92210d0d2187a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f15b1d759ec8dc197ce8fb6976916b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_ee09fc693aaed9fc21c4bfdc8c404ce2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36e2b43c39aa331caad51c7f85841170(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee09fc693aaed9fc21c4bfdc8c404ce2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_4c897aa5e08d8f0516f51b0822c07d0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[217413], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cca607aa7642821e26eac3afd4569fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c897aa5e08d8f0516f51b0822c07d0b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_0a833ece4e05f497dc02cd776756a5ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6c7912337582331187d46c416807cc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a833ece4e05f497dc02cd776756a5ab
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_93de34a1e06a18d89ab119f6c791d44b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_370a3f2bb072cd9cbf26a01afcaae5e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93de34a1e06a18d89ab119f6c791d44b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_d1dc75780dcc07695e144eafa6ccb22b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8631da6b4df25bbf26c84b3807e4065f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1dc75780dcc07695e144eafa6ccb22b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_2527f7a0a2065d512461c5dd30699270(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89ee7ab4c1b3b6d1ccd673af82e5285e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2527f7a0a2065d512461c5dd30699270
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_89ee7ab4c1b3b6d1ccd673af82e5285e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2527f7a0a2065d512461c5dd30699270
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2ac6b2391d15e4f2252cd76e92d8b5b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e491f245cb708b93db7b16c0ebde4582
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_25b15ae027d48e447546907f0a61b847(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b19a09d0f1c9b07b47cfbbeff5ba671
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d6c7912337582331187d46c416807cc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a833ece4e05f497dc02cd776756a5ab
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5a28c4cf9b50742c7d78bbfab1c2efba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_552a6d7ceb14455ee72b6b494fb4d219
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_370a3f2bb072cd9cbf26a01afcaae5e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93de34a1e06a18d89ab119f6c791d44b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9ab586694a24fc6cc92210d0d2187a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f15b1d759ec8dc197ce8fb6976916b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_76d2448490720937830a1a823ef6c934(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[185658], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae48b10ed6a45e4a48b1f20ff33c3fce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76d2448490720937830a1a823ef6c934
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10113a70e745e56e1ff2d452e807e793(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8068aa076a8f63200d75976a1d2c234f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2ac6b2391d15e4f2252cd76e92d8b5b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e491f245cb708b93db7b16c0ebde4582
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1a7cca2ebc4e917e2e81680f43ceddc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_478c7e17f9568a938ce46a7c8b0b83f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[150], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f8d839cbdb2c01a35795e873827a6100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e76d6c8c9bf76e5c4afd4cfcd36a9be0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_462d17dc2d9ba74bac96ad52e605c162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1a7cca2ebc4e917e2e81680f43ceddc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6f4eb50a69ce7d2077189aec7e2b2190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_605285c7b3d6e85d4cd54e5144d0753e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_21ed57a976ba8ec90dc089f244e6ed11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_da942ad6246c2e0bd681fdfe77623903(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.greater_equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d75698c5768e61cacf57f95a87bc75a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_90b537f2b33d865928bb11aff3a69ac6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3f5a8bd20235d88e92e5bc31aa72fecb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_91b13fbd4673ef723c37cb27075c39ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_91b13fbd4673ef723c37cb27075c39ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[15200], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d75698c5768e61cacf57f95a87bc75a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.009999999776482582], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_87a41ad7988d27bbc9d377c75e241ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_87a41ad7988d27bbc9d377c75e241ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cc90450adf11622c6f9cd96f11fa116b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_100c045f7d05c65734cd7d2bb85f57e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_22eb895348efd19228ccb67d1c70f57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2204], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6162465b11661a89f607030f3c00a5f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_45948fdca2dc3b4bf5444313466a75a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9fc82a179a328912b06c8cac475c65b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d4188bf1878030ca5177e317dbeabc3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[551], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_134399d0256b527981bedbd745445f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2135d500adee5c985c0c282ebbf6d743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7271bcd12b0c26ca0c5f66dbb533bcc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f749b8cf6feb3dbfaec8e4f1246d5e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[8816], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_775bd5825dd06d433c1de8e9308874b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_775bd5825dd06d433c1de8e9308874b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cc90450adf11622c6f9cd96f11fa116b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_87a41ad7988d27bbc9d377c75e241ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2135d500adee5c985c0c282ebbf6d743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[247], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1a7cca2ebc4e917e2e81680f43ceddc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3800], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7271bcd12b0c26ca0c5f66dbb533bcc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[950], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9fc82a179a328912b06c8cac475c65b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6dc17bab7c968c7d4b0c578d1c16de8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[70], dtype='int64'),
                paddle.to_tensor(0, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_39dc98983b3b04ea258dd8fe0e5db3d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca68f5e07c29a517581f8c3006902ff8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_85ed02eb4b055b60393d33e6cf4ba4af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cc90450adf11622c6f9cd96f11fa116b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da942ad6246c2e0bd681fdfe77623903
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.10000000149011612], dtype='float32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()