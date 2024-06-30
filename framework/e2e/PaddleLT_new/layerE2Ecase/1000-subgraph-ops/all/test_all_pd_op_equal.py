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
    class PrimitiveOp_5d9b341d6bf37918762475ac6182a704(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d1c1eac1071d60d6c5bc6f9608b0a32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e766e8034f71ac2edfc75cbdbd675e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6ed9d33005ddd29e209a9bef486ac86f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_056449f0ca1d6852fad8b56cd59153f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0550dd5c7932699453b446dd24b4e775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_07e79cda6a3e2beee5d0e81b53bc11bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_beb82c51265ae16567f82e0f15549bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f6d4752785b1bbe98574462b90378a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2bdb9e0836dffeddec0bb729dd8bbf62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3812093679b2ee5a084c11b778ad7dde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3f9f39bdfd355e57af634470cb9347db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_104ce7c03966eecbcddc3fdfd931b1a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.equal(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de0c2c90964d0bd7b5f18464670e9999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_104ce7c03966eecbcddc3fdfd931b1a2
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3572ad43aab3b4d0be10f66cfcfbacfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_356253733d340579260982019be6ea5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5704cffa5c017c2021b84937fd82397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5f4459f8a3e363823214119c4781ec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_73cd5deffcd9422ac99bfe77f31d639f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8dd81ba9ce36c3eed49bc15f61c06190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c587a07542a44ec0ba2dba5289a85abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_48fb79924ca1adcf1ba1b8c895ea9771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4d1c1eac1071d60d6c5bc6f9608b0a32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[86970], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e766e8034f71ac2edfc75cbdbd675e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[242991], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6ed9d33005ddd29e209a9bef486ac86f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[220968], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_056449f0ca1d6852fad8b56cd59153f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[153450], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0550dd5c7932699453b446dd24b4e775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_07e79cda6a3e2beee5d0e81b53bc11bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_beb82c51265ae16567f82e0f15549bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185691], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f6d4752785b1bbe98574462b90378a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2bdb9e0836dffeddec0bb729dd8bbf62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3812093679b2ee5a084c11b778ad7dde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3f9f39bdfd355e57af634470cb9347db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_de0c2c90964d0bd7b5f18464670e9999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_104ce7c03966eecbcddc3fdfd931b1a2
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3572ad43aab3b4d0be10f66cfcfbacfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[113061], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_356253733d340579260982019be6ea5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[205923], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5704cffa5c017c2021b84937fd82397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[123783], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5f4459f8a3e363823214119c4781ec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[171888], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_73cd5deffcd9422ac99bfe77f31d639f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[217413], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8dd81ba9ce36c3eed49bc15f61c06190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c587a07542a44ec0ba2dba5289a85abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_48fb79924ca1adcf1ba1b8c895ea9771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d9b341d6bf37918762475ac6182a704
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[185658], dtype='int32'),
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()