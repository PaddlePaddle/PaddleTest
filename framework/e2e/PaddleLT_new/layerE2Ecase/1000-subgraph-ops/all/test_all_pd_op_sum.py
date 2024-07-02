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
    class PrimitiveOp_b2d71599066af3ee2b334e3860f2b204(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccf96f32ef509da5a50734c0993dd271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2d71599066af3ee2b334e3860f2b204
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce4d3be02fc5ea7d91b01a980cd9e2be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([4296], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4b2f3d94c67893f9acb2af7b7ec6013(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49ded85d15757b818c71ae02812e51e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b2f3d94c67893f9acb2af7b7ec6013
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85405468c97fc749f0d777956393b07d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6859bc3e4e39db5d12698c8bdba11bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cbcb759516a6c15902497cb45bf844d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cbcb759516a6c15902497cb45bf844d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa631bb3b99961fda87c5114e76a4e95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.01621961034834385, 0.010803415440022945]], [[0.00011908607848454267, 0.002841575536876917]], [[0.002326482906937599, 0.04811672121286392]], [[0.07740908861160278, 0.1652306318283081]], [[0.016101844608783722, 0.1074819564819336]], [[0.03895648941397667, 0.0001602049742359668]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_6cc918822c37b075f146ae506b0db532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0048958840779960155, 0.006318248808383942]], [[0.0745432898402214, 0.006702129263430834]], [[0.014964740723371506, 0.057525549083948135]], [[0.0033520697616040707, 0.023395076394081116]], [[0.012890883721411228, 0.02136208489537239]], [[0.032561663538217545, 0.001504171290434897]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    
    class PrimitiveOp_967950939eba7293178c26d71e2605f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dba9b2d6a7399a9afd5c4fae0d8c1c32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_967950939eba7293178c26d71e2605f5
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b12b6f005cb24a444b2f1717274ec13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.19400350749492645, 0.1547604352235794, 0.13279034197330475, 0.25442981719970703, 0.03932170569896698, 0.15875673294067383, 0.047717031091451645, 0.24486544728279114, 0.10118933767080307, 0.050853170454502106, 0.019079243764281273, 0.2431994080543518, 0.18871864676475525, 0.1056072860956192, 0.040733274072408676, 0.050507787615060806], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_d37d7a63dbaad4dc5a525df369111173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a3e3b86c53cf3247ca74d4491c94e78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_647a432068a0d62fcd0f5bc66bad7688(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ec1cdc36c39888863cc7896596acf53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf7db718c8efc0ec3d2bc56de49dee40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a818642c743a73617856280c0bb672cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a818642c743a73617856280c0bb672cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_916dbdad7643dbe51346c348180ff6f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cd5ded4fea307bd826a67f6e7e0c57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05614259093999863, 0.2648021876811981, 0.14120104908943176, 0.11532451212406158], [0.0013617724180221558, 0.029156655073165894, 0.20148354768753052, 0.27942192554473877], [0.10520064830780029, 0.225392147898674, 0.20918087661266327, 0.17586283385753632], [0.3509211242198944, 0.3742552399635315, 0.25012320280075073, 0.011851891875267029], [0.033196449279785156, 0.07336187362670898, 0.22557489573955536, 0.398066908121109]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_8ce85620e49fcce1e3418278ee3e7c58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_267d764b38d50334eb8389e371ac28e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ce85620e49fcce1e3418278ee3e7c58
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_988804713af27c54402e2b1babca2631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.030529730021953583, 0.2831009328365326, 0.36617031693458557, 0.04184707999229431], [0.07069867849349976, 0.2227173000574112, 0.041081346571445465, 0.03687387704849243], [0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767], [0.07069867849349976, 0.2227173000574112, 0.041081346571445465, 0.03687387704849243], [0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_98015b238d33e14d24523e5ac01bbf8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b2f3d94c67893f9acb2af7b7ec6013
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82b1c318ca10b41622728a3a96782f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ab891b3fada3f9bee48f1c0148ea3c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ab891b3fada3f9bee48f1c0148ea3c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66419ff22aa1262bf5b0c2badefad281(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.027252674102783203, 0.39509862661361694, 0.02363678812980652, 0.1274213194847107], [0.20206919312477112, 0.032816849648952484, 0.32369446754455566, 0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [0.20206919312477112, 0.032816849648952484, 0.32369446754455566, 0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [0.3455130159854889, 0.1529913991689682, 0.24695870280265808, 0.2722965180873871], [0.3455130159854889, 0.1529913991689682, 0.24695870280265808, 0.2722965180873871]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_438b0106fab5ed7839ac708c537e79de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_46961cbcfff12cca688b16efbfcfd20c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8dc2d0780e6ec7bb7077cc4e5245ae81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46961cbcfff12cca688b16efbfcfd20c
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bfd057a86bc34310e85f172ff58dbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7db718c8efc0ec3d2bc56de49dee40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06fd7844ca545ce9c2e53aad174994c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06fd7844ca545ce9c2e53aad174994c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_916dbdad7643dbe51346c348180ff6f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7527193b9b5d1b12362ae39993ee3173(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e74f75ab6cebeb0cd9c9333cfac534b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7527193b9b5d1b12362ae39993ee3173
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0bf561c40e0cc552ae3227adda06c5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.22009597718715668, 0.1568000316619873, 0.08095628023147583, 0.22521916031837463, 0.1262054294347763, 0.18052004277706146, 0.16559453308582306, 0.19467540085315704, 0.1783711314201355, 0.05074217915534973, 0.11262712627649307, 0.23964127898216248, 0.10786125063896179, 0.04298098385334015, 0.2244531512260437, 0.16581885516643524, 0.07022041082382202, 0.14405877888202667, 0.26927465200424194, 0.030548853799700737, 0.16125410795211792, 0.20930293202400208, 0.046851422637701035, 0.182158425450325], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b87028a2060cf8a12fc4f5487034643e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74e3498697fddd1a717b072362bf9838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74e3498697fddd1a717b072362bf9838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00a52a694832cbc40f0216a4e2e05337(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b606e7dbdecdc0785ced46c43c60745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a52a694832cbc40f0216a4e2e05337
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f72b7dabaf4fbb835f84229abd2aaa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2500355541706085, 0.07697464525699615, 0.2121511548757553, 0.06575275957584381], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93fd2ebe500b0aeb23a6f9ab1cfcf03d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2679559886455536, 0.09963895380496979, 0.050239741802215576, 0.09699589759111404], [0.4907260537147522, 0.0988239049911499, 0.043211743235588074, 0.13195684552192688], [0.22280314564704895, 0.2555414140224457, 0.20873220264911652, 0.1573859304189682], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, 0.022008880972862244], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, 0.022008880972862244], [0.22280314564704895, 0.2555414140224457, 0.20873220264911652, 0.1573859304189682]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aea951e001fbd71554280adb35a7fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42347490787506104, 0.44859740138053894, 0.025512784719467163, 0.1642150580883026], [0.38556623458862305, 0.3216448426246643, 0.11168976873159409, 0.03786981850862503], [0.0014653801918029785, 0.13820701837539673, 0.07339861989021301, 0.0377705842256546], [0.3172072768211365, 0.15113165974617004, 0.22106342017650604, 0.08921520411968231], [0.42347490787506104, 0.44859740138053894, 0.025512784719467163, 0.1642150580883026]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5c1a782cdad9843efb5fce7822def8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b606e7dbdecdc0785ced46c43c60745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a52a694832cbc40f0216a4e2e05337
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcfb2e9e0c60f8e1589c8c2b54ea8da0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03413599729537964, 0.28878089785575867, 0.3469504117965698, 0.19062592089176178], [0.2298850268125534, 0.05640196055173874, 0.03873452544212341, 0.22728106379508972], [0.02583347260951996, 0.3125740587711334, 0.10351571440696716, 0.242221400141716], [0.33769890666007996, 0.06839853525161743, 0.11585550010204315, 0.06547404825687408]], dtype='float32').reshape([4, 4]),
            ]


    
    class PrimitiveOp_caa2a3793f6320ee128769ce60a35efb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7561a0ea689e342e2585e3a2b04a9ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_caa2a3793f6320ee128769ce60a35efb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74f75ab6cebeb0cd9c9333cfac534b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7527193b9b5d1b12362ae39993ee3173
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b00467828dfc58e49efffcff276af55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39358b168757d2d5f9f3c7a4c820c31d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeb863e79e0a7a3e8d992ed7d7f08c32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9b4589c41fe3094316da5f94600c262(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8b762bbc278656eec17b5efd2fd7320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8b762bbc278656eec17b5efd2fd7320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb9d52d1290d0d22b57f2b9b1e66955f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_caa2a3793f6320ee128769ce60a35efb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96ce8d03ae930f34e5c6ebe7ddc9aae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.140933558344841, 0.00036913156509399414, 0.33528417348861694, 0.08211620897054672], [0.140933558344841, 0.00036913156509399414, 0.33528417348861694, 0.08211620897054672], [0.1097898781299591, 0.3261854350566864, 0.044446952641010284, 0.019107185304164886], [0.20964205265045166, 0.009211540222167969, 0.2750697135925293, 0.14300581812858582], [0.06909973919391632, 0.23634669184684753, 0.23610402643680573, 0.2775801420211792], [0.30173084139823914, 0.3142993748188019, 0.1273236721754074, 0.06189805269241333], [0.09281511604785919, 0.13114047050476074, 0.019350245594978333, 0.03895244002342224]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_d1ae2640c2aecd65a038cc7178c3a299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01aa67b105f53bbd1c8a13cb9e0cead5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01aa67b105f53bbd1c8a13cb9e0cead5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0ddcc3c5cbef5db61d0e746363a7ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([4830], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74f0eb5e24387c1ec368ce3dc15bfdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([1199], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48cac42ea15e3e47c6bea9d703dc2bcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b2f3d94c67893f9acb2af7b7ec6013
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1dd0f3bdff689697153e6ed20789496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ca28edf7fd91bddbb74a6cc0f209c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70ca28edf7fd91bddbb74a6cc0f209c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28249fb18ebe5e6c5bf9a289fb082cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07022564113140106, 0.24063928425312042, 0.2388477772474289, 0.0949905663728714], [0.03909817337989807, 0.15637561678886414, 0.20350569486618042, 0.2634051442146301], [0.03909817337989807, 0.15637561678886414, 0.20350569486618042, 0.2634051442146301], [0.06787297129631042, 0.008341282606124878, 0.10706061124801636, 0.25277483463287354], [0.29549020528793335, 0.19191592931747437, 0.07872748374938965, 0.09537695348262787], [0.10860984027385712, 0.09693586826324463, 0.16498376429080963, 0.18836523592472076]], dtype='float32').reshape([6, 4]),
            ]


    
    class PrimitiveOp_ff02e37b3d5ab248ffdd188aeb998224(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_656c8ad27c70dad5b178db9613d6e8a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff02e37b3d5ab248ffdd188aeb998224
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d849d182909eb31bfa43b40872a70638(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b9fea08fcdc7417837599047264cac7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d849d182909eb31bfa43b40872a70638
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_002d55acde146fd636efcb11c50d4fce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61ec819876d728444a1fbec6c686b743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_002d55acde146fd636efcb11c50d4fce
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd8d4407e29803843162cc7b43298277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0a729538886d2036257cfb3fef4a817(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0a729538886d2036257cfb3fef4a817(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_503bf8f1fe5d0f0fb7e9e006ebaa6e39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ec6a23b90b3dae5e66b37822c5c6cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ec6a23b90b3dae5e66b37822c5c6cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ba3e9efd9a3d73a399b8e607b8e84fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d61417011faa03c1cd38998846c0192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d61417011faa03c1cd38998846c0192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08afaf3049f84c4466382e1a8bc13b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_652ae5e361f4d8114246c20b5337cc24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0d8ddb9275e62446445b4e8fefe86c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_652ae5e361f4d8114246c20b5337cc24
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_beb3ac352cb6c1e35980565398c778cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e10d08a89cae95b203a92ec23d21c523(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beb3ac352cb6c1e35980565398c778cd
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dc2d0780e6ec7bb7077cc4e5245ae81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46961cbcfff12cca688b16efbfcfd20c
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff1be6683f4669c53fd1b508903ec119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09198623895645142, 0.226093128323555, 0.08336149156093597, 0.21535493433475494, 0.14010421931743622, 0.04213462769985199, 0.13986985385417938, 0.10578273236751556, 0.06852646917104721, 0.022478558123111725, 0.08370701223611832, 0.049561627209186554, 0.2402431219816208, 0.049292754381895065, 0.13204483687877655, 0.11585280299186707, 0.23965030908584595, 0.014008288271725178, 0.03199032321572304, 0.2534511089324951], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_39df4bb536637fa547004a6412dc72a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([17475], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0d8ddb9275e62446445b4e8fefe86c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_652ae5e361f4d8114246c20b5337cc24
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d637982f018c4a95e34d134f743b7a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047e5a3a6fbb01ee662945d9be931a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fbd6719e355f93f7873c4d7e9b427d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a24225ed1c31e25ab0cf108250a4646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a24225ed1c31e25ab0cf108250a4646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4bf94170fbaf5210c41a43d0758e4615(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91d9ed3d3c6ebb2e4a77276ecce38979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bf94170fbaf5210c41a43d0758e4615
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e9f4d7253bda179727e0cfc75d71e3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_379075d362fd82b563efd5afed7acab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06421732902526855, 0.04781962186098099, 0.16483047604560852, 0.16644680500030518], [0.10803155601024628, 0.03807184100151062, 0.24309112131595612, 0.014126971364021301], [0.1445263922214508, 0.45514774322509766, 0.1134442538022995, 0.06528478860855103], [0.1445263922214508, 0.45514774322509766, 0.1134442538022995, 0.06528478860855103], [0.34093427658081055, 0.1452919840812683, 0.3238971531391144, 0.25240251421928406]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_0b9fea08fcdc7417837599047264cac7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d849d182909eb31bfa43b40872a70638
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b6cb051c54955cb9b462f77c92f44e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d241fce2d713aa280d592cda7dd6062f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24557a4f91169bbf40ed000b85e80f39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1a476a4f0053392ebbbbafaff3eb135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29a12e704cf3cc3a679b6c85f5093cd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29a12e704cf3cc3a679b6c85f5093cd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc3a653e5c52b05ff2267cb887dee22a
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37d2f175119659e7ba8d740cf32654d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1356925368309021, 0.3836618661880493, 0.2497148960828781, 0.10594353079795837], [0.13692587614059448, 0.18728870153427124, 0.11343744397163391, 0.3271639943122864], [0.17433902621269226, 0.08971483260393143, 0.13315066695213318, 0.015453487634658813], [0.1356925368309021, 0.3836618661880493, 0.2497148960828781, 0.10594353079795837], [0.11636693775653839, 0.08864608407020569, 0.22304943203926086, 0.282735139131546], [0.14897876977920532, 0.186306431889534, 0.06858530640602112, 0.24515454471111298], [0.11636693775653839, 0.08864608407020569, 0.22304943203926086, 0.282735139131546]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_5559426af0f86d18bfa7e690563c4614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e10d08a89cae95b203a92ec23d21c523(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_beb3ac352cb6c1e35980565398c778cd
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_663b7a1569ddc8890933fe87888afc86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4dd6bad8c7c8bf1b103c904388cb30f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce4d3be02fc5ea7d91b01a980cd9e2be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([4296], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49ded85d15757b818c71ae02812e51e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b2f3d94c67893f9acb2af7b7ec6013
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6859bc3e4e39db5d12698c8bdba11bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cbcb759516a6c15902497cb45bf844d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cbcb759516a6c15902497cb45bf844d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa631bb3b99961fda87c5114e76a4e95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.01621961034834385, 0.010803415440022945]], [[0.00011908607848454267, 0.002841575536876917]], [[0.002326482906937599, 0.04811672121286392]], [[0.07740908861160278, 0.1652306318283081]], [[0.016101844608783722, 0.1074819564819336]], [[0.03895648941397667, 0.0001602049742359668]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_6cc918822c37b075f146ae506b0db532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50358e31d2bb3e9e0eec49970e2e1fcc
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0048958840779960155, 0.006318248808383942]], [[0.0745432898402214, 0.006702129263430834]], [[0.014964740723371506, 0.057525549083948135]], [[0.0033520697616040707, 0.023395076394081116]], [[0.012890883721411228, 0.02136208489537239]], [[0.032561663538217545, 0.001504171290434897]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_34f9e1a5f8997f53f317d313501090ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b12b6f005cb24a444b2f1717274ec13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.19400350749492645, 0.1547604352235794, 0.13279034197330475, 0.25442981719970703, 0.03932170569896698, 0.15875673294067383, 0.047717031091451645, 0.24486544728279114, 0.10118933767080307, 0.050853170454502106, 0.019079243764281273, 0.2431994080543518, 0.18871864676475525, 0.1056072860956192, 0.040733274072408676, 0.050507787615060806], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_d37d7a63dbaad4dc5a525df369111173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a3e3b86c53cf3247ca74d4491c94e78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_647a432068a0d62fcd0f5bc66bad7688(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7db718c8efc0ec3d2bc56de49dee40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89cf7a9c715645a7bf28baa294991b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89cf7a9c715645a7bf28baa294991b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_916dbdad7643dbe51346c348180ff6f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cd5ded4fea307bd826a67f6e7e0c57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05614259093999863, 0.2648021876811981, 0.14120104908943176, 0.11532451212406158], [0.0013617724180221558, 0.029156655073165894, 0.20148354768753052, 0.27942192554473877], [0.10520064830780029, 0.225392147898674, 0.20918087661266327, 0.17586283385753632], [0.3509211242198944, 0.3742552399635315, 0.25012320280075073, 0.011851891875267029], [0.033196449279785156, 0.07336187362670898, 0.22557489573955536, 0.398066908121109]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_a6e84fe2da44315042fb12219803a60c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_988804713af27c54402e2b1babca2631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.030529730021953583, 0.2831009328365326, 0.36617031693458557, 0.04184707999229431], [0.07069867849349976, 0.2227173000574112, 0.041081346571445465, 0.03687387704849243], [0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767], [0.07069867849349976, 0.2227173000574112, 0.041081346571445465, 0.03687387704849243], [0.38079363107681274, 0.0006191432476043701, 0.3535638451576233, 0.09530767798423767]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_98015b238d33e14d24523e5ac01bbf8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b2f3d94c67893f9acb2af7b7ec6013
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82b1c318ca10b41622728a3a96782f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5381d1fc7fe5b886f4727f1a3d0a6f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5381d1fc7fe5b886f4727f1a3d0a6f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66419ff22aa1262bf5b0c2badefad281(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.027252674102783203, 0.39509862661361694, 0.02363678812980652, 0.1274213194847107], [0.20206919312477112, 0.032816849648952484, 0.32369446754455566, 0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [0.20206919312477112, 0.032816849648952484, 0.32369446754455566, 0.09777773916721344], [0.2741281986236572, 0.16037379205226898, 0.44714775681495667, 0.0011175870895385742], [0.3455130159854889, 0.1529913991689682, 0.24695870280265808, 0.2722965180873871], [0.3455130159854889, 0.1529913991689682, 0.24695870280265808, 0.2722965180873871]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_438b0106fab5ed7839ac708c537e79de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91f88075caff7228fe99147545b8822e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02bfd057a86bc34310e85f172ff58dbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7db718c8efc0ec3d2bc56de49dee40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9be74d7192919df125786ffbcc154df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9be74d7192919df125786ffbcc154df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_916dbdad7643dbe51346c348180ff6f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab360bda54b6deed1ca201634b6d0e82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0bf561c40e0cc552ae3227adda06c5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.22009597718715668, 0.1568000316619873, 0.08095628023147583, 0.22521916031837463, 0.1262054294347763, 0.18052004277706146, 0.16559453308582306, 0.19467540085315704, 0.1783711314201355, 0.05074217915534973, 0.11262712627649307, 0.23964127898216248, 0.10786125063896179, 0.04298098385334015, 0.2244531512260437, 0.16581885516643524, 0.07022041082382202, 0.14405877888202667, 0.26927465200424194, 0.030548853799700737, 0.16125410795211792, 0.20930293202400208, 0.046851422637701035, 0.182158425450325], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b87028a2060cf8a12fc4f5487034643e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df3c74bdb1de28a50fae6899b7d67b55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df3c74bdb1de28a50fae6899b7d67b55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e99a72a45ef87d4d5310ca1f376e101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f72b7dabaf4fbb835f84229abd2aaa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2500355541706085, 0.07697464525699615, 0.2121511548757553, 0.06575275957584381], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93fd2ebe500b0aeb23a6f9ab1cfcf03d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2679559886455536, 0.09963895380496979, 0.050239741802215576, 0.09699589759111404], [0.4907260537147522, 0.0988239049911499, 0.043211743235588074, 0.13195684552192688], [0.22280314564704895, 0.2555414140224457, 0.20873220264911652, 0.1573859304189682], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, 0.022008880972862244], [0.0817495584487915, 0.20968347787857056, 0.3012875020503998, 0.022008880972862244], [0.22280314564704895, 0.2555414140224457, 0.20873220264911652, 0.1573859304189682]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aea951e001fbd71554280adb35a7fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42347490787506104, 0.44859740138053894, 0.025512784719467163, 0.1642150580883026], [0.38556623458862305, 0.3216448426246643, 0.11168976873159409, 0.03786981850862503], [0.0014653801918029785, 0.13820701837539673, 0.07339861989021301, 0.0377705842256546], [0.3172072768211365, 0.15113165974617004, 0.22106342017650604, 0.08921520411968231], [0.42347490787506104, 0.44859740138053894, 0.025512784719467163, 0.1642150580883026]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5c1a782cdad9843efb5fce7822def8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e99a72a45ef87d4d5310ca1f376e101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcfb2e9e0c60f8e1589c8c2b54ea8da0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03413599729537964, 0.28878089785575867, 0.3469504117965698, 0.19062592089176178], [0.2298850268125534, 0.05640196055173874, 0.03873452544212341, 0.22728106379508972], [0.02583347260951996, 0.3125740587711334, 0.10351571440696716, 0.242221400141716], [0.33769890666007996, 0.06839853525161743, 0.11585550010204315, 0.06547404825687408]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_f7561a0ea689e342e2585e3a2b04a9ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_caa2a3793f6320ee128769ce60a35efb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab360bda54b6deed1ca201634b6d0e82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b00467828dfc58e49efffcff276af55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39358b168757d2d5f9f3c7a4c820c31d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeb863e79e0a7a3e8d992ed7d7f08c32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9b4589c41fe3094316da5f94600c262(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41e860c7690e9fec517b70e946aef0ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41e860c7690e9fec517b70e946aef0ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb9d52d1290d0d22b57f2b9b1e66955f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_caa2a3793f6320ee128769ce60a35efb
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96ce8d03ae930f34e5c6ebe7ddc9aae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.140933558344841, 0.00036913156509399414, 0.33528417348861694, 0.08211620897054672], [0.140933558344841, 0.00036913156509399414, 0.33528417348861694, 0.08211620897054672], [0.1097898781299591, 0.3261854350566864, 0.044446952641010284, 0.019107185304164886], [0.20964205265045166, 0.009211540222167969, 0.2750697135925293, 0.14300581812858582], [0.06909973919391632, 0.23634669184684753, 0.23610402643680573, 0.2775801420211792], [0.30173084139823914, 0.3142993748188019, 0.1273236721754074, 0.06189805269241333], [0.09281511604785919, 0.13114047050476074, 0.019350245594978333, 0.03895244002342224]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_d1ae2640c2aecd65a038cc7178c3a299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0843e939c111f85ee4848d3ae0d267df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0843e939c111f85ee4848d3ae0d267df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0ddcc3c5cbef5db61d0e746363a7ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([4830], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74f0eb5e24387c1ec368ce3dc15bfdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([1199], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48cac42ea15e3e47c6bea9d703dc2bcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b2f3d94c67893f9acb2af7b7ec6013
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1dd0f3bdff689697153e6ed20789496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_377cfb1458f579e89e36a5e3b4697fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_377cfb1458f579e89e36a5e3b4697fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28249fb18ebe5e6c5bf9a289fb082cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07022564113140106, 0.24063928425312042, 0.2388477772474289, 0.0949905663728714], [0.03909817337989807, 0.15637561678886414, 0.20350569486618042, 0.2634051442146301], [0.03909817337989807, 0.15637561678886414, 0.20350569486618042, 0.2634051442146301], [0.06787297129631042, 0.008341282606124878, 0.10706061124801636, 0.25277483463287354], [0.29549020528793335, 0.19191592931747437, 0.07872748374938965, 0.09537695348262787], [0.10860984027385712, 0.09693586826324463, 0.16498376429080963, 0.18836523592472076]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_4c8c2b2a150af19968810f25e4c33e4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23b606673f2a19d3173efb4febbcfdbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc2b334f2cc38a45737d30fc15eac8e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd8d4407e29803843162cc7b43298277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_717a46b4784bb551c868df3d58c14c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_717a46b4784bb551c868df3d58c14c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_503bf8f1fe5d0f0fb7e9e006ebaa6e39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3dab08bbc8235971c1082840a62650f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3dab08bbc8235971c1082840a62650f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ba3e9efd9a3d73a399b8e607b8e84fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1c6d59df439ea672c6dbb66b37b5ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1c6d59df439ea672c6dbb66b37b5ef2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08afaf3049f84c4466382e1a8bc13b44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d9bc240992ba584b267f313f336f6d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3277f9fea9a49fade2fe2e268eb96c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91f88075caff7228fe99147545b8822e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff1be6683f4669c53fd1b508903ec119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09198623895645142, 0.226093128323555, 0.08336149156093597, 0.21535493433475494, 0.14010421931743622, 0.04213462769985199, 0.13986985385417938, 0.10578273236751556, 0.06852646917104721, 0.022478558123111725, 0.08370701223611832, 0.049561627209186554, 0.2402431219816208, 0.049292754381895065, 0.13204483687877655, 0.11585280299186707, 0.23965030908584595, 0.014008288271725178, 0.03199032321572304, 0.2534511089324951], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_39df4bb536637fa547004a6412dc72a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([17475], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d9bc240992ba584b267f313f336f6d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d637982f018c4a95e34d134f743b7a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047e5a3a6fbb01ee662945d9be931a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fbd6719e355f93f7873c4d7e9b427d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5c832bfd975d4ad781c8fda0aa83fed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5c832bfd975d4ad781c8fda0aa83fed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7e76419485a96a64bdf58447ea1c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e9f4d7253bda179727e0cfc75d71e3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_379075d362fd82b563efd5afed7acab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06421732902526855, 0.04781962186098099, 0.16483047604560852, 0.16644680500030518], [0.10803155601024628, 0.03807184100151062, 0.24309112131595612, 0.014126971364021301], [0.1445263922214508, 0.45514774322509766, 0.1134442538022995, 0.06528478860855103], [0.1445263922214508, 0.45514774322509766, 0.1134442538022995, 0.06528478860855103], [0.34093427658081055, 0.1452919840812683, 0.3238971531391144, 0.25240251421928406]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_23b606673f2a19d3173efb4febbcfdbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91b6cb051c54955cb9b462f77c92f44e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d241fce2d713aa280d592cda7dd6062f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24557a4f91169bbf40ed000b85e80f39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1a476a4f0053392ebbbbafaff3eb135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ec1cdc36c39888863cc7896596acf53
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2aebaa5806775e68dcd46ea8f8bf22e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2aebaa5806775e68dcd46ea8f8bf22e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5362fe81728ec89d8b255a42acab2498(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f691a05dbf34bc3d7a17824721e52e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37d2f175119659e7ba8d740cf32654d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1356925368309021, 0.3836618661880493, 0.2497148960828781, 0.10594353079795837], [0.13692587614059448, 0.18728870153427124, 0.11343744397163391, 0.3271639943122864], [0.17433902621269226, 0.08971483260393143, 0.13315066695213318, 0.015453487634658813], [0.1356925368309021, 0.3836618661880493, 0.2497148960828781, 0.10594353079795837], [0.11636693775653839, 0.08864608407020569, 0.22304943203926086, 0.282735139131546], [0.14897876977920532, 0.186306431889534, 0.06858530640602112, 0.24515454471111298], [0.11636693775653839, 0.08864608407020569, 0.22304943203926086, 0.282735139131546]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_5559426af0f86d18bfa7e690563c4614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85405468c97fc749f0d777956393b07d
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3277f9fea9a49fade2fe2e268eb96c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_663b7a1569ddc8890933fe87888afc86
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()